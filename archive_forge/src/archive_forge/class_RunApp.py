from __future__ import annotations
import queue
import signal
import sys
import time
import typing as t
from jupyter_core.application import JupyterApp, base_aliases, base_flags
from traitlets import Any, Dict, Float
from traitlets.config import catch_config_error
from . import __version__
from .consoleapp import JupyterConsoleApp, app_aliases, app_flags
class RunApp(JupyterApp, JupyterConsoleApp):
    """An Jupyter Console app to run files."""
    version = __version__
    name = 'jupyter run'
    description = 'Run Jupyter kernel code.'
    flags = Dict(flags)
    aliases = Dict(aliases)
    frontend_aliases = Any(frontend_aliases)
    frontend_flags = Any(frontend_flags)
    kernel_timeout = Float(60, config=True, help='Timeout for giving up on a kernel (in seconds).\n\n        On first connect and restart, the console tests whether the\n        kernel is running and responsive by sending kernel_info_requests.\n        This sets the timeout in seconds for how long the kernel can take\n        before being presumed dead.\n        ')

    def parse_command_line(self, argv: list[str] | None=None) -> None:
        """Parse the command line arguments."""
        super().parse_command_line(argv)
        self.build_kernel_argv(self.extra_args)
        self.filenames_to_run = self.extra_args[:]

    @catch_config_error
    def initialize(self, argv: list[str] | None=None) -> None:
        """Initialize the app."""
        self.log.debug('jupyter run: initialize...')
        super().initialize(argv)
        JupyterConsoleApp.initialize(self)
        signal.signal(signal.SIGINT, self.handle_sigint)
        self.init_kernel_info()

    def handle_sigint(self, *args: t.Any) -> None:
        """Handle SIGINT."""
        if self.kernel_manager:
            self.kernel_manager.interrupt_kernel()
        else:
            self.log.error("Cannot interrupt kernels we didn't start.\n")

    def init_kernel_info(self) -> None:
        """Wait for a kernel to be ready, and store kernel info"""
        timeout = self.kernel_timeout
        tic = time.time()
        self.kernel_client.hb_channel.unpause()
        msg_id = self.kernel_client.kernel_info()
        while True:
            try:
                reply = self.kernel_client.get_shell_msg(timeout=1)
            except queue.Empty as e:
                if time.time() - tic > timeout:
                    msg = "Kernel didn't respond to kernel_info_request"
                    raise RuntimeError(msg) from e
            else:
                if reply['parent_header'].get('msg_id') == msg_id:
                    self.kernel_info = reply['content']
                    return

    def start(self) -> None:
        """Start the application."""
        self.log.debug('jupyter run: starting...')
        super().start()
        if self.filenames_to_run:
            for filename in self.filenames_to_run:
                self.log.debug('jupyter run: executing `%s`', filename)
                with open(filename) as fp:
                    code = fp.read()
                    reply = self.kernel_client.execute_interactive(code, timeout=OUTPUT_TIMEOUT)
                    return_code = 0 if reply['content']['status'] == 'ok' else 1
                    if return_code:
                        raise Exception("jupyter-run error running '%s'" % filename)
        else:
            code = sys.stdin.read()
            reply = self.kernel_client.execute_interactive(code, timeout=OUTPUT_TIMEOUT)
            return_code = 0 if reply['content']['status'] == 'ok' else 1
            if return_code:
                msg = "jupyter-run error running 'stdin'"
                raise Exception(msg)