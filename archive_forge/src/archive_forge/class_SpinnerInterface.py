import contextlib
import itertools
import logging
import sys
import time
from typing import IO, Generator, Optional
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.logging import get_indentation
class SpinnerInterface:

    def spin(self) -> None:
        raise NotImplementedError()

    def finish(self, final_status: str) -> None:
        raise NotImplementedError()