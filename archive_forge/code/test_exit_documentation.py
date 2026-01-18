from io import StringIO
from typing import Optional, Union
import twisted.trial.unittest
from ...runner import _exit
from .._exit import ExitStatus, exit

        L{exit} given a non-zero status code writes the given message to
        standard error.
        