import atexit
import functools
import locale
import logging
import multiprocessing
import os
import traceback
import pathlib
import Pyro4.core
import argparse
from enum import IntEnum
import shutil
import socket
import struct
import collections
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
import uuid
import psutil
import Pyro4
from random import Random
from minerl.env import comms
import minerl.utils.process_watcher
@Pyro4.expose
class MinecraftInstance(object):
    """
    A subprocess wrapper which maintains a reference to a minecraft subprocess
    and also allows for stable closing and launching of such subprocesses 
    across different platforms.

    The Minecraft instance class works by launching two subprocesses:
    the Malmo subprocess, and a watcher subprocess with access to 
    the process IDs of both the parent process and the Malmo subprocess.
    If the parent process dies, it will kill the subprocess, and then itself.

    This scheme has a single failure point of the process dying before the watcher process is launched.
    """
    MAX_PIPE_LENGTH = 500

    def __init__(self, port=None, existing=False, status_dir=None, seed=None, instance_id=None, max_mem=None):
        """
        Launches the subprocess.

        Note: max_mem should be a string following the same format as Java's -Xmx flag (e.g. 4G = 4 gigs of max mem).
        """
        self.running = False
        self._starting = True
        self.minecraft_process = None
        self.watcher_process = None
        self.xml = None
        self.role = None
        self.client_socket = None
        self._port = port
        self._host = InstanceManager.DEFAULT_IP
        self.locked = False
        self.uuid = str(uuid.uuid4()).replace('-', '')[:6]
        self.existing = existing
        self.minecraft_dir = None
        self.instance_dir = None
        self._status_dir = status_dir
        self.owner = None
        self._max_mem = max_mem
        self.instance_id = instance_id
        try:
            seed = InstanceManager._get_next_seed(instance_id)
        except TypeError as e:
            pass
        finally:
            self._seed = seed
        self._setup_logging()
        self._target_port = port

    @property
    def actor_name(self):
        return f'actor{self.role}'

    def launch(self, daemonize=False, replaceable=True):
        port = self._target_port
        self._starting = True
        if not self.existing:
            if not port:
                port = InstanceManager._get_valid_port()
            self.minecraft_dir = InstanceManager.MINECRAFT_DIR
            self.instance_dir = os.path.join(InstanceManager.MINECRAFT_DIR, '..')
            parent_pid = os.getpid()
            self.minecraft_process = self._launch_minecraft(port, InstanceManager.headless, self.minecraft_dir, replaceable=replaceable)
            if not daemonize:
                self.watcher_process = minerl.utils.process_watcher.launch(parent_pid, self.minecraft_process.pid, self.instance_dir)
            lines = []
            client_ready = False
            server_ready = False
            while True:
                mine_log_encoding = locale.getpreferredencoding(False)
                line = self.minecraft_process.stdout.readline().decode(mine_log_encoding)
                _check_for_launch_errors(line)
                if not line:
                    error_str = ''
                    for l in lines:
                        spline = '\n'.join(l.split('\n')[:-1])
                        self._logger.error(spline)
                        error_str += spline + '\n'
                    raise EOFError(error_str + '\n\nMinecraft process finished unexpectedly. There was an error with Malmo.')
                lines.append(line)
                self._log_heuristic('\n'.join(line.split('\n')[:-1]))
                MALMOENVPORTSTR = '***** Start MalmoEnvServer on port '
                port_received = MALMOENVPORTSTR in line
                if port_received:
                    self._port = int(line.split(MALMOENVPORTSTR)[-1].strip())
                client_ready = 'CLIENT enter state: DORMANT' in line
                server_ready = 'SERVER enter state: DORMANT' in line
                if client_ready:
                    break
            if not self.port:
                raise RuntimeError('Malmo failed to start the MalmoEnv server! Check the logs from the Minecraft process.')
            self._logger.info('Minecraft process ready')
            if not port == self._port:
                self._logger.warning('Tried to launch Minecraft on port {} but that port was taken, instead Minecraft is using port {}.'.format(port, self.port))

            def log_to_file(logdir):
                if not os.path.exists(os.path.join(logdir, 'logs')):
                    os.makedirs(os.path.join(logdir, 'logs'))
                file_path = os.path.join(logdir, 'logs', 'mc_{}.log'.format(self._target_port - 9000))
                logger.info('Logging output of Minecraft to {}'.format(file_path))
                mine_log = open(file_path, 'wb+')
                mine_log.truncate(0)
                mine_log_encoding = locale.getpreferredencoding(False)
                try:
                    while self.running:
                        line = self.minecraft_process.stdout.readline()
                        if not line:
                            break
                        try:
                            linestr = line.decode(mine_log_encoding)
                        except UnicodeDecodeError:
                            mine_log_encoding = locale.getpreferredencoding(False)
                            logger.error('UnicodeDecodeError, switching to default encoding')
                            linestr = line.decode(mine_log_encoding)
                        linestr = '\n'.join(linestr.split('\n')[:-1])
                        self._log_heuristic(linestr)
                        mine_log.write(line)
                        mine_log.flush()
                finally:
                    mine_log.close()
            logdir = os.environ.get('MALMO_MINECRAFT_OUTPUT_LOGDIR', '.')
            self._logger_thread = threading.Thread(target=functools.partial(log_to_file, logdir=logdir))
            self._logger_thread.setDaemon(True)
            self._logger_thread.start()
        else:
            assert port is not None, 'No existing port specified.'
            self._port = port
        self.running = True
        self._starting = False
        if not daemonize:
            atexit.register(lambda: self._destruct())

    def kill(self):
        """
        Kills the process (if it has been launched.)
        """
        self._destruct()
        pass

    def close(self):
        """Closes the object.
        """
        self._destruct(should_close=True)

    @property
    def status_dir(self):
        return self._status_dir

    @property
    def host(self):
        return self._host

    @property
    def port(self):
        return self._port

    def get_output(self):
        while self.running or self._starting:
            try:
                level, line = self._output_stream.pop()
                return ((line.levelno, line.getMessage(), line.name), self.running or self._starting)
            except IndexError:
                time.sleep(0.1)
        else:
            return (None, False)

    def _setup_logging(self):
        self._logger = logging.getLogger(__name__ + '.instance.{}'.format(str(self.uuid)))
        self._output_stream = collections.deque(maxlen=self.MAX_PIPE_LENGTH)
        for level in [logging.DEBUG]:
            handler = comms.QueueLogger(self._output_stream)
            handler.setLevel(level)
            self._logger.addHandler(handler)

    def _launch_minecraft(self, port, headless, minecraft_dir, replaceable=False):
        """Launch Minecraft listening for malmoenv connections.
        Args:
            port:  the TCP port to listen on.
            installdir: the install dir name. Defaults to MalmoPlatform.
            Must be same as given (or defaulted) in download call if used.
            replaceable: whether or not to automatically restart Minecraft (default is false).
                         Does not work on Windows.
        Asserts:
            that the port specified is open.
        """
        launch_script = 'launchClient.sh'
        if os.name == 'nt':
            launch_script = 'launchClient.bat'
        launch_script = os.path.join(minecraft_dir, launch_script)
        rundir = os.path.join(minecraft_dir, 'run')
        cmd = [launch_script, '-port', str(port), '-env', '-runDir', rundir]
        if self.status_dir:
            cmd += ['-performanceDir', self.status_dir]
        if self._seed:
            cmd += ['-seed', ','.join([str(x) for x in self._seed])]
        if self._max_mem:
            cmd += ['-maxMem', self._max_mem]
        cmd_to_print = cmd[:] if not self._seed else cmd[:-2]
        self._logger.info('Starting Minecraft process: ' + str(cmd_to_print))
        if replaceable:
            cmd.append('-replaceable')
        preexec_fn = os.setsid if 'linux' in str(sys.platform) or sys.platform == 'darwin' else None
        minecraft_process = psutil.Popen(cmd, cwd=InstanceManager.MINECRAFT_DIR, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, preexec_fn=preexec_fn)
        return minecraft_process

    @staticmethod
    def _kill_minecraft_via_malmoenv(host, port):
        """Use carefully to cause the Minecraft service to exit (and hopefully restart).
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            sock.connect((host, port))
            comms.send_message(sock, ('<MalmoEnv' + malmo_version + '/>').encode())
            comms.send_message(sock, '<Exit>NOW</Exit>'.encode())
            reply = comms.recv_message(sock)
            ok, = struct.unpack('!I', reply)
            sock.close()
            return ok == 1
        except Exception as e:
            logger.error('Attempted to send kill command to minecraft process and failed with exception {}'.format(e))
            return False

    def __del__(self):
        """
        On destruction of this instance kill the child.
        """
        self._destruct()

    def _destruct(self, should_close=False):
        """
        Do our best as the parent process to destruct and kill the child + watcher.
        """
        if (self.running or should_close) and (not self.existing):
            self.running = False
            self._starting = False
            time.sleep(1)
            if self._kill_minecraft_via_malmoenv(self.host, self.port):
                time.sleep(2)
            try:
                minerl.utils.process_watcher.reap_process_and_children(self.minecraft_process)
            except psutil.NoSuchProcess:
                pass
            if self in InstanceManager._instance_pool:
                InstanceManager._instance_pool.remove(self)
                self.release_lock()
        pass

    def __repr__(self):
        return 'Malmo[{}:{}, proc={}, addr={}:{}, locked={}]'.format(self.role, self.uuid, self.minecraft_process.pid if not self.existing else 'EXISTING', self.host, self.port, self.locked)

    def _acquire_lock(self, owner=None):
        self.locked = True
        self.owner = owner

    def release_lock(self):
        self.locked = False
        self.owner = None

    def _log_heuristic(self, msg):
        """
        Log the message, heuristically determine logging level based on the
        message content
        """
        if ('STDERR' in msg or 'ERROR' in msg or 'Exception' in msg or ('    at ' in msg) or msg.startswith('Error')) and (not 'connection closed, likely by peer' in msg):
            self._logger.error(msg)
        elif 'WARN' in msg:
            self._logger.warn(msg)
        elif 'LOGTOPY' in msg:
            self._logger.info(msg)
        else:
            self._logger.debug(msg)