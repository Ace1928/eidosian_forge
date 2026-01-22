from __future__ import annotations
import getopt
import inspect
import os
import sys
import textwrap
from os import path
from typing import Any, Dict, Optional, cast
from twisted.python import reflect, util
@property
def _repeatFlag(self):
    if self._repeat:
        return '*'
    else:
        return ''