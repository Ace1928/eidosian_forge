import sys
from contextlib import (
from typing import (
from .utils import (  # namedtuple_with_defaults,

        Provide functionality to call application commands by calling PyBridge
        ex: app('help')
        :param command: command line being run
        :param echo: If provided, this temporarily overrides the value of self.cmd_echo while the
                     command runs. If True, output will be echoed to stdout/stderr. (Defaults to None)

        