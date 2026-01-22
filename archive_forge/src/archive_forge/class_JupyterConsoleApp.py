import atexit
import os
import signal
import sys
import typing as t
import uuid
import warnings
from jupyter_core.application import base_aliases, base_flags
from traitlets import CBool, CUnicode, Dict, List, Type, Unicode
from traitlets.config.application import boolean_flag
from . import KernelManager, connect, find_connection_file, tunnel_to_kernel
from .blocking import BlockingKernelClient
from .connect import KernelConnectionInfo
from .kernelspec import NoSuchKernel
from .localinterfaces import localhost
from .restarter import KernelRestarter
from .session import Session
from .utils import _filefind
class JupyterConsoleApp(ConnectionFileMixin):
    """The base Jupyter console application."""
    name: t.Union[str, Unicode] = 'jupyter-console-mixin'
    description: t.Union[str, Unicode] = '\n        The Jupyter Console Mixin.\n\n        This class contains the common portions of console client (QtConsole,\n        ZMQ-based terminal console, etc).  It is not a full console, in that\n        launched terminal subprocesses will not be able to accept input.\n\n        The Console using this mixing supports various extra features beyond\n        the single-process Terminal IPython shell, such as connecting to\n        existing kernel, via:\n\n            jupyter console <appname> --existing\n\n        as well as tunnel via SSH\n\n    '
    classes = classes
    flags = Dict(flags)
    aliases = Dict(aliases)
    kernel_manager_class = Type(default_value=KernelManager, config=True, help='The kernel manager class to use.')
    kernel_client_class = BlockingKernelClient
    kernel_argv = List(Unicode())
    sshserver = Unicode('', config=True, help='The SSH server to use to connect to the kernel.')
    sshkey = Unicode('', config=True, help='Path to the ssh key to use for logging in to the ssh server.')

    def _connection_file_default(self) -> str:
        return 'kernel-%i.json' % os.getpid()
    existing = CUnicode('', config=True, help='Connect to an already running kernel')
    kernel_name = Unicode('python', config=True, help='The name of the default kernel to start.')
    confirm_exit = CBool(True, config=True, help="\n        Set to display confirmation dialog on exit. You can always use 'exit' or 'quit',\n        to force a direct exit without any confirmation.")

    def build_kernel_argv(self, argv: object=None) -> None:
        """build argv to be passed to kernel subprocess

        Override in subclasses if any args should be passed to the kernel
        """
        self.kernel_argv = self.extra_args

    def init_connection_file(self) -> None:
        """find the connection file, and load the info if found.

        The current working directory and the current profile's security
        directory will be searched for the file if it is not given by
        absolute path.

        When attempting to connect to an existing kernel and the `--existing`
        argument does not match an existing file, it will be interpreted as a
        fileglob, and the matching file in the current profile's security dir
        with the latest access time will be used.

        After this method is called, self.connection_file contains the *full path*
        to the connection file, never just its name.
        """
        runtime_dir = self.runtime_dir
        if self.existing:
            try:
                cf = find_connection_file(self.existing, ['.', runtime_dir])
            except Exception:
                self.log.critical('Could not find existing kernel connection file %s', self.existing)
                self.exit(1)
            self.log.debug('Connecting to existing kernel: %s', cf)
            self.connection_file = cf
        else:
            try:
                cf = find_connection_file(self.connection_file, [runtime_dir])
            except Exception:
                if self.connection_file == os.path.basename(self.connection_file):
                    cf = os.path.join(runtime_dir, self.connection_file)
                else:
                    cf = self.connection_file
                self.connection_file = cf
        try:
            self.connection_file = _filefind(self.connection_file, ['.', runtime_dir])
        except OSError:
            self.log.debug('Connection File not found: %s', self.connection_file)
            return
        try:
            self.load_connection_file()
        except Exception:
            self.log.error('Failed to load connection file: %r', self.connection_file, exc_info=True)
            self.exit(1)

    def init_ssh(self) -> None:
        """set up ssh tunnels, if needed."""
        if not self.existing or (not self.sshserver and (not self.sshkey)):
            return
        self.load_connection_file()
        transport = self.transport
        ip = self.ip
        if transport != 'tcp':
            self.log.error('Can only use ssh tunnels with TCP sockets, not %s', transport)
            sys.exit(-1)
        if self.sshkey and (not self.sshserver):
            self.sshserver = ip
            ip = localhost()
        info: KernelConnectionInfo = {'ip': ip, 'shell_port': self.shell_port, 'iopub_port': self.iopub_port, 'stdin_port': self.stdin_port, 'hb_port': self.hb_port, 'control_port': self.control_port}
        self.log.info('Forwarding connections to %s via %s', ip, self.sshserver)
        self.ip = localhost()
        try:
            newports = tunnel_to_kernel(info, self.sshserver, self.sshkey)
        except:
            self.log.error('Could not setup tunnels', exc_info=True)
            self.exit(1)
        self.shell_port, self.iopub_port, self.stdin_port, self.hb_port, self.control_port = newports
        cf = self.connection_file
        root, ext = os.path.splitext(cf)
        self.connection_file = root + '-ssh' + ext
        self.write_connection_file()
        self.log.info('To connect another client via this tunnel, use:')
        self.log.info('--existing %s', os.path.basename(self.connection_file))

    def _new_connection_file(self) -> str:
        cf = ''
        while not cf:
            ident = str(uuid.uuid4()).split('-')[-1]
            runtime_dir = self.runtime_dir
            cf = os.path.join(runtime_dir, 'kernel-%s.json' % ident)
            cf = cf if not os.path.exists(cf) else ''
        return cf

    def init_kernel_manager(self) -> None:
        """Initialize the kernel manager."""
        if self.existing:
            self.kernel_manager = None
            return
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        try:
            self.kernel_manager = self.kernel_manager_class(ip=self.ip, session=self.session, transport=self.transport, shell_port=self.shell_port, iopub_port=self.iopub_port, stdin_port=self.stdin_port, hb_port=self.hb_port, control_port=self.control_port, connection_file=self.connection_file, kernel_name=self.kernel_name, parent=self, data_dir=self.data_dir)
        except NoSuchKernel:
            self.log.critical('Could not find kernel %s', self.kernel_name)
            self.exit(1)
        self.kernel_manager = t.cast(KernelManager, self.kernel_manager)
        self.kernel_manager.client_factory = self.kernel_client_class
        kwargs = {}
        kwargs['extra_arguments'] = self.kernel_argv
        self.kernel_manager.start_kernel(**kwargs)
        atexit.register(self.kernel_manager.cleanup_ipc_files)
        if self.sshserver:
            self.kernel_manager.write_connection_file()
        km = self.kernel_manager
        self.shell_port = km.shell_port
        self.iopub_port = km.iopub_port
        self.stdin_port = km.stdin_port
        self.hb_port = km.hb_port
        self.control_port = km.control_port
        self.connection_file = km.connection_file
        atexit.register(self.kernel_manager.cleanup_connection_file)

    def init_kernel_client(self) -> None:
        """Initialize the kernel client."""
        if self.kernel_manager is not None:
            self.kernel_client = self.kernel_manager.client()
        else:
            self.kernel_client = self.kernel_client_class(session=self.session, ip=self.ip, transport=self.transport, shell_port=self.shell_port, iopub_port=self.iopub_port, stdin_port=self.stdin_port, hb_port=self.hb_port, control_port=self.control_port, connection_file=self.connection_file, parent=self)
        self.kernel_client.start_channels()

    def initialize(self, argv: object=None) -> None:
        """
        Classes which mix this class in should call:
               JupyterConsoleApp.initialize(self,argv)
        """
        if getattr(self, '_dispatching', False):
            return
        self.init_connection_file()
        self.init_ssh()
        self.init_kernel_manager()
        self.init_kernel_client()