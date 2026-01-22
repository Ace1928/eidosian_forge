import os
import sys
import warnings
from pathlib import Path
from threading import local
from IPython.core import page, payloadpage
from IPython.core.autocall import ZMQExitAutocall
from IPython.core.displaypub import DisplayPublisher
from IPython.core.error import UsageError
from IPython.core.interactiveshell import InteractiveShell, InteractiveShellABC
from IPython.core.magic import Magics, line_magic, magics_class
from IPython.core.magics import CodeMagics, MacroToEdit  # type:ignore[attr-defined]
from IPython.core.usage import default_banner
from IPython.display import Javascript, display
from IPython.utils import openpy
from IPython.utils.process import arg_split, system  # type:ignore[attr-defined]
from jupyter_client.session import Session, extract_header
from jupyter_core.paths import jupyter_runtime_dir
from traitlets import Any, CBool, CBytes, Dict, Instance, Type, default, observe
from ipykernel import connect_qtconsole, get_connection_file, get_connection_info
from ipykernel.displayhook import ZMQShellDisplayHook
from ipykernel.jsonutil import encode_images, json_clean
class ZMQInteractiveShell(InteractiveShell):
    """A subclass of InteractiveShell for ZMQ."""
    displayhook_class = Type(ZMQShellDisplayHook)
    display_pub_class = Type(ZMQDisplayPublisher)
    data_pub_class = Any()
    kernel = Any()
    parent_header = Any()

    @default('banner1')
    def _default_banner1(self):
        return default_banner
    readline_use = CBool(False)
    autoindent = CBool(False)
    exiter = Instance(ZMQExitAutocall)

    @default('exiter')
    def _default_exiter(self):
        return ZMQExitAutocall(self)

    @observe('exit_now')
    def _update_exit_now(self, change):
        """stop eventloop when exit_now fires"""
        if change['new']:
            if hasattr(self.kernel, 'io_loop'):
                loop = self.kernel.io_loop
                loop.call_later(0.1, loop.stop)
            if self.kernel.eventloop:
                exit_hook = getattr(self.kernel.eventloop, 'exit_hook', None)
                if exit_hook:
                    exit_hook(self.kernel)
    keepkernel_on_exit = None

    def enable_gui(self, gui):
        """Enable a given guil."""
        from .eventloops import enable_gui as real_enable_gui
        try:
            real_enable_gui(gui)
            self.active_eventloop = gui
        except ValueError as e:
            raise UsageError('%s' % e) from e

    def init_environment(self):
        """Configure the user's environment."""
        env = os.environ
        env['TERM'] = 'xterm-color'
        env['CLICOLOR'] = '1'
        env['FORCE_COLOR'] = '1'
        env['CLICOLOR_FORCE'] = '1'
        env['PAGER'] = 'cat'
        env['GIT_PAGER'] = 'cat'

    def init_hooks(self):
        """Initialize hooks."""
        super().init_hooks()
        self.set_hook('show_in_pager', page.as_hook(payloadpage.page), 99)

    def init_data_pub(self):
        """Delay datapub init until request, for deprecation warnings"""

    @property
    def data_pub(self):
        if not hasattr(self, '_data_pub'):
            warnings.warn('InteractiveShell.data_pub is deprecated outside IPython parallel.', DeprecationWarning, stacklevel=2)
            self._data_pub = self.data_pub_class(parent=self)
            self._data_pub.session = self.display_pub.session
            self._data_pub.pub_socket = self.display_pub.pub_socket
        return self._data_pub

    @data_pub.setter
    def data_pub(self, pub):
        self._data_pub = pub

    def ask_exit(self):
        """Engage the exit actions."""
        self.exit_now = not self.keepkernel_on_exit
        payload = dict(source='ask_exit', keepkernel=self.keepkernel_on_exit)
        self.payload_manager.write_payload(payload)

    def run_cell(self, *args, **kwargs):
        """Run a cell."""
        self._last_traceback = None
        return super().run_cell(*args, **kwargs)

    def _showtraceback(self, etype, evalue, stb):
        sys.stdout.flush()
        sys.stderr.flush()
        exc_content = {'traceback': stb, 'ename': str(etype.__name__), 'evalue': str(evalue)}
        dh = self.displayhook
        topic = None
        if dh.topic:
            topic = dh.topic.replace(b'execute_result', b'error')
        dh.session.send(dh.pub_socket, 'error', json_clean(exc_content), dh.parent_header, ident=topic)
        self._last_traceback = stb

    def set_next_input(self, text, replace=False):
        """Send the specified text to the frontend to be presented at the next
        input cell."""
        payload = dict(source='set_next_input', text=text, replace=replace)
        self.payload_manager.write_payload(payload)

    def set_parent(self, parent):
        """Set the parent header for associating output with its triggering input"""
        self.parent_header = parent
        self.displayhook.set_parent(parent)
        self.display_pub.set_parent(parent)
        if hasattr(self, '_data_pub'):
            self.data_pub.set_parent(parent)
        try:
            sys.stdout.set_parent(parent)
        except AttributeError:
            pass
        try:
            sys.stderr.set_parent(parent)
        except AttributeError:
            pass

    def get_parent(self):
        """Get the parent header."""
        return self.parent_header

    def init_magics(self):
        """Initialize magics."""
        super().init_magics()
        self.register_magics(KernelMagics)
        self.magics_manager.register_alias('ed', 'edit')

    def init_virtualenv(self):
        """Initialize virtual environment."""

    def system_piped(self, cmd):
        """Call the given cmd in a subprocess, piping stdout/err

        Parameters
        ----------
        cmd : str
            Command to execute (can not end in '&', as background processes are
            not supported.  Should not be a command that expects input
            other than simple text.
        """
        if cmd.rstrip().endswith('&'):
            msg = 'Background processes not supported.'
            raise OSError(msg)
        if sys.platform == 'win32':
            cmd = self.var_expand(cmd, depth=1)
            from IPython.utils._process_win32 import AvoidUNCPath
            with AvoidUNCPath() as path:
                if path is not None:
                    cmd = f'pushd {path} &&{cmd}'
                self.user_ns['_exit_code'] = system(cmd)
        else:
            self.user_ns['_exit_code'] = system(self.var_expand(cmd, depth=1))
    system = system_piped