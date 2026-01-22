from __future__ import annotations
import getopt
import inspect
import os
import sys
import textwrap
from os import path
from typing import Any, Dict, Optional, cast
from twisted.python import reflect, util
class CompleteUsernames(Completer):
    """
    Complete usernames
    """

    def _shellCode(self, optName, shellType):
        if shellType == _ZSH:
            return f'{self._repeatFlag}:{self._description(optName)}:_users'
        raise NotImplementedError(f'Unknown shellType {shellType!r}')