import os
import re
import time
import logging
import warnings
import subprocess
from typing import List, Type, Tuple, Union, Optional, cast
from os.path import join as pjoin
from os.path import split as psplit
from libcloud.utils.py3 import StringIO, b
from libcloud.utils.logging import ExtraLogFormatter
class ParamikoSSHClient(BaseSSHClient):
    """
    A SSH Client powered by Paramiko.
    """
    CHUNK_SIZE = 4096
    SLEEP_DELAY = 0.2

    def __init__(self, hostname, port=22, username='root', password=None, key=None, key_files=None, key_material=None, timeout=None, keep_alive=None, use_compression=False):
        """
        Authentication is always attempted in the following order:

        - The key passed in (if key is provided)
        - Any key we can find through an SSH agent (only if no password and
          key is provided)
        - Any "id_rsa" or "id_dsa" key discoverable in ~/.ssh/ (only if no
          password and key is provided)
        - Plain username/password auth, if a password was given (if password is
          provided)

        :param keep_alive: Optional keep alive internal (in seconds) to use.
        :type keep_alive: ``int``

        :param use_compression: True to use compression.
        :type use_compression: ``bool``
        """
        if key_files and key_material:
            raise ValueError('key_files and key_material arguments are mutually exclusive')
        super().__init__(hostname=hostname, port=port, username=username, password=password, key=key, key_files=key_files, timeout=timeout)
        self.key_material = key_material
        self.keep_alive = keep_alive
        self.use_compression = use_compression
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.logger = self._get_and_setup_logger()
        self.sftp_client = None

    def connect(self):
        conninfo = {'hostname': self.hostname, 'port': self.port, 'username': self.username, 'allow_agent': False, 'look_for_keys': False}
        if self.password:
            conninfo['password'] = self.password
        if self.key_files:
            conninfo['key_filename'] = self.key_files
        if self.key_material:
            conninfo['pkey'] = self._get_pkey_object(key=self.key_material, password=self.password)
        if not self.password and (not (self.key_files or self.key_material)):
            conninfo['allow_agent'] = True
            conninfo['look_for_keys'] = True
        if self.timeout:
            conninfo['timeout'] = self.timeout
        if self.key_files and (not isinstance(self.key_files, (list, tuple))) and os.path.isfile(self.key_files):
            with open(self.key_files) as fp:
                key_material = fp.read()
            try:
                pkey = self._get_pkey_object(key=key_material, password=self.password)
            except paramiko.ssh_exception.PasswordRequiredException as e:
                raise e
            except Exception:
                pass
            else:
                del conninfo['key_filename']
                conninfo['pkey'] = pkey
        extra = {'_hostname': self.hostname, '_port': self.port, '_username': self.username, '_timeout': self.timeout}
        if self.password:
            extra['_auth_method'] = 'password'
        else:
            extra['_auth_method'] = 'key_file'
            if self.key_files:
                extra['_key_file'] = self.key_files
        self.logger.debug('Connecting to server', extra=extra)
        try:
            self.client.connect(**conninfo)
        except paramiko.ssh_exception.AuthenticationException as e:
            if PARAMIKO_VERSION_TUPLE >= (2, 9, 0) and LIBCLOUD_PARAMIKO_SHA2_BACKWARD_COMPATIBILITY:
                self.logger.warn(SHA2_PUBKEY_NOT_SUPPORTED_AUTH_ERROR_MSG)
                conninfo['disabled_algorithms'] = {'pubkeys': ['rsa-sha2-256', 'rsa-sha2-512']}
                self.client.connect(**conninfo)
            else:
                raise e
        return True

    def put(self, path, contents=None, chmod=None, mode='w'):
        extra = {'_path': path, '_mode': mode, '_chmod': chmod}
        self.logger.debug('Uploading file', extra=extra)
        sftp = self._get_sftp_client()
        head, tail = psplit(path)
        if path[0] == '/':
            sftp.chdir('/')
        else:
            sftp.chdir('.')
        for part in head.split('/'):
            if part != '':
                try:
                    sftp.mkdir(part)
                except OSError:
                    pass
                sftp.chdir(part)
        cwd = sftp.getcwd()
        cwd = self._sanitize_cwd(cwd=cwd)
        ak = sftp.file(tail, mode=mode)
        ak.write(contents)
        if chmod is not None:
            ak.chmod(chmod)
        ak.close()
        file_path = self._sanitize_file_path(cwd=cwd, file_path=path)
        return file_path

    def putfo(self, path, fo=None, chmod=None):
        """
        Upload file like object to the remote server.

        Unlike put(), this method operates on file objects and not directly on
        file content which makes it much more efficient for large files since
        it utilizes pipelining.
        """
        extra = {'_path': path, '_chmod': chmod}
        self.logger.debug('Uploading file', extra=extra)
        sftp = self._get_sftp_client()
        head, tail = psplit(path)
        if path[0] == '/':
            sftp.chdir('/')
        else:
            sftp.chdir('.')
        for part in head.split('/'):
            if part != '':
                try:
                    sftp.mkdir(part)
                except OSError:
                    pass
                sftp.chdir(part)
        cwd = sftp.getcwd()
        cwd = self._sanitize_cwd(cwd=cwd)
        sftp.putfo(fo, path)
        if chmod is not None:
            ak = sftp.file(tail)
            ak.chmod(chmod)
            ak.close()
        file_path = self._sanitize_file_path(cwd=cwd, file_path=path)
        return file_path

    def delete(self, path):
        extra = {'_path': path}
        self.logger.debug('Deleting file', extra=extra)
        sftp = self.client.open_sftp()
        sftp.unlink(path)
        sftp.close()
        return True

    def run(self, cmd, timeout=None):
        """
        Note: This function is based on paramiko's exec_command()
        method.

        :param timeout: How long to wait (in seconds) for the command to
                        finish (optional).
        :type timeout: ``float``
        """
        extra1 = {'_cmd': cmd}
        self.logger.debug('Executing command', extra=extra1)
        bufsize = -1
        transport = self._get_transport()
        chan = transport.open_session()
        start_time = time.time()
        chan.exec_command(cmd)
        stdout = StringIO()
        stderr = StringIO()
        stdin = chan.makefile('wb', bufsize)
        stdin.close()
        exit_status_ready = chan.exit_status_ready()
        if exit_status_ready:
            stdout.write(self._consume_stdout(chan).getvalue())
            stderr.write(self._consume_stderr(chan).getvalue())
        while not exit_status_ready:
            current_time = time.time()
            elapsed_time = current_time - start_time
            if timeout and elapsed_time > timeout:
                chan.close()
                stdout_str = stdout.getvalue()
                stderr_str = stderr.getvalue()
                raise SSHCommandTimeoutError(cmd=cmd, timeout=timeout, stdout=stdout_str, stderr=stderr_str)
            stdout.write(self._consume_stdout(chan).getvalue())
            stderr.write(self._consume_stderr(chan).getvalue())
            exit_status_ready = chan.exit_status_ready()
            if exit_status_ready:
                break
            time.sleep(self.SLEEP_DELAY)
        status = chan.recv_exit_status()
        stdout_str = stdout.getvalue()
        stderr_str = stderr.getvalue()
        extra2 = {'_status': status, '_stdout': stdout_str, '_stderr': stderr_str}
        self.logger.debug('Command finished', extra=extra2)
        result = (stdout_str, stderr_str, status)
        return result

    def close(self):
        self.logger.debug('Closing server connection')
        if self.client:
            self.client.close()
        if self.sftp_client:
            self.sftp_client.close()
        return True

    def _consume_stdout(self, chan):
        """
        Try to consume stdout data from chan if it's receive ready.
        """
        stdout = self._consume_data_from_channel(chan=chan, recv_method=chan.recv, recv_ready_method=chan.recv_ready)
        return stdout

    def _consume_stderr(self, chan):
        """
        Try to consume stderr data from chan if it's receive ready.
        """
        stderr = self._consume_data_from_channel(chan=chan, recv_method=chan.recv_stderr, recv_ready_method=chan.recv_stderr_ready)
        return stderr

    def _consume_data_from_channel(self, chan, recv_method, recv_ready_method):
        """
        Try to consume data from the provided channel.

        Keep in mind that data is only consumed if the channel is receive
        ready.
        """
        result = StringIO()
        result_bytes = bytearray()
        if recv_ready_method():
            data = recv_method(self.CHUNK_SIZE)
            result_bytes += b(data)
            while data:
                ready = recv_ready_method()
                if not ready:
                    break
                data = recv_method(self.CHUNK_SIZE)
                result_bytes += b(data)
        result.write(result_bytes.decode('utf-8', errors='ignore'))
        return result

    def _get_pkey_object(self, key, password=None):
        """
        Try to detect private key type and return paramiko.PKey object.

        # NOTE: Paramiko only supports key in PKCS#1 PEM format.
        """
        key_types = [(paramiko.RSAKey, 'RSA'), (paramiko.DSSKey, 'DSA'), (paramiko.ECDSAKey, 'EC')]
        paramiko_version = getattr(paramiko, '__version__', '0.0.0')
        paramiko_version = tuple((int(c) for c in paramiko_version.split('.')))
        if paramiko_version >= (2, 2, 0):
            key_types.append((paramiko.ed25519key.Ed25519Key, 'Ed25519'))
        for cls, key_type in key_types:
            key_split = key.strip().splitlines()
            if key_split[0] == '-----BEGIN PRIVATE KEY-----' and key_split[-1] == '-----END PRIVATE KEY-----':
                key_split[0] = '-----BEGIN %s PRIVATE KEY-----' % key_type
                key_split[-1] = '-----END %s PRIVATE KEY-----' % key_type
                key_value = '\n'.join(key_split)
            else:
                key_value = key
            try:
                key = cls.from_private_key(StringIO(key_value), password)
            except paramiko.ssh_exception.PasswordRequiredException as e:
                raise e
            except (paramiko.ssh_exception.SSHException, AssertionError) as e:
                if 'private key file checkints do not match' in str(e).lower():
                    msg = 'Invalid password provided for encrypted key. Original error: %s' % str(e)
                    raise paramiko.ssh_exception.SSHException(msg)
                pass
            else:
                return key
        msg = 'Invalid or unsupported key type (only RSA, DSS, ECDSA and Ed25519 keys in PEM format are supported). For more information on  supported key file types, see %s' % SUPPORTED_KEY_TYPES_URL
        raise paramiko.ssh_exception.SSHException(msg)

    def _sanitize_cwd(self, cwd):
        if re.match('^\\/\\w\\:.*$', str(cwd)):
            cwd = str(cwd[1:])
        return cwd

    def _sanitize_file_path(self, cwd, file_path):
        """
        Sanitize the provided file path and ensure we always return an
        absolute path, even if relative path is passed to to this function.
        """
        if file_path[0] in ['/', '\\'] or re.match('^\\w\\:.*$', file_path):
            pass
        elif re.match('^\\w\\:.*$', cwd):
            file_path = cwd + '\\' + file_path
        else:
            file_path = pjoin(cwd, file_path)
        return file_path

    def _get_transport(self):
        """
        Return transport object taking into account keep alive and compression
        options passed to the constructor.
        """
        transport = self.client.get_transport()
        if self.keep_alive:
            transport.set_keepalive(self.keep_alive)
        if self.use_compression:
            transport.use_compression(compress=True)
        return transport

    def _get_sftp_client(self):
        """
        Create SFTP client from the underlying SSH client.

        This method tries to re-use the existing self.sftp_client (if it
        exists) and it also tries to verify the connection is opened and if
        it's not, it will try to re-establish it.
        """
        if not self.sftp_client:
            self.sftp_client = self.client.open_sftp()
        sftp_client = self.sftp_client
        try:
            sftp_client.listdir('.')
        except OSError as e:
            if 'socket is closed' in str(e).lower():
                self.sftp_client = self.client.open_sftp()
            elif 'no such file' in str(e).lower():
                pass
            else:
                raise e
        return self.sftp_client