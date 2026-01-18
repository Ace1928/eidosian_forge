from argparse import Action
import os
import sys
from typing import Generator
from typing import List
from typing import Optional
from typing import Union
from _pytest.config import Config
from _pytest.config import ExitCode
from _pytest.config import PrintHelp
from _pytest.config.argparsing import Parser
from _pytest.terminal import TerminalReporter
import pytest
An argparse Action that will raise an exception in order to skip the
    rest of the argument parsing when --help is passed.

    This prevents argparse from quitting due to missing required arguments
    when any are defined, for example by ``pytest_addoption``.
    This is similar to the way that the builtin argparse --help option is
    implemented by raising SystemExit.
    