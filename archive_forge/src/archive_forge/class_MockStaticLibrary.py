from __future__ import annotations
import os
import sys
import typing as T
from .. import mparser, mesonlib
from .. import environment
from ..interpreterbase import (
from ..interpreter import (
from ..mparser import (
class MockStaticLibrary(MesonInterpreterObject):
    pass