import sys
from contextlib import (
from typing import (
from .utils import (  # namedtuple_with_defaults,
class PyBridge:
    """Provides a Python API wrapper for application commands."""

    def __init__(self, cmd2_app: 'cmd2.Cmd') -> None:
        self._cmd2_app = cmd2_app
        self.cmd_echo = False
        self.stop = False

    def __dir__(self) -> List[str]:
        """Return a custom set of attribute names"""
        attributes: List[str] = []
        attributes.insert(0, 'cmd_echo')
        return attributes

    def __call__(self, command: str, *, echo: Optional[bool]=None) -> CommandResult:
        """
        Provide functionality to call application commands by calling PyBridge
        ex: app('help')
        :param command: command line being run
        :param echo: If provided, this temporarily overrides the value of self.cmd_echo while the
                     command runs. If True, output will be echoed to stdout/stderr. (Defaults to None)

        """
        if echo is None:
            echo = self.cmd_echo
        copy_cmd_stdout = StdSim(cast(Union[TextIO, StdSim], self._cmd2_app.stdout), echo=echo)
        copy_cmd_stdout.pause_storage = True
        copy_stderr = StdSim(sys.stderr, echo=echo)
        self._cmd2_app.last_result = None
        stop = False
        try:
            self._cmd2_app.stdout = cast(TextIO, copy_cmd_stdout)
            with redirect_stdout(cast(IO[str], copy_cmd_stdout)):
                with redirect_stderr(cast(IO[str], copy_stderr)):
                    stop = self._cmd2_app.onecmd_plus_hooks(command, py_bridge_call=True)
        finally:
            with self._cmd2_app.sigint_protection:
                self._cmd2_app.stdout = cast(IO[str], copy_cmd_stdout.inner_stream)
                self.stop = stop or self.stop
        result = CommandResult(stdout=copy_cmd_stdout.getvalue(), stderr=copy_stderr.getvalue(), stop=stop, data=self._cmd2_app.last_result)
        return result