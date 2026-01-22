from __future__ import annotations
import os
from fnmatch import fnmatch
from typing import (
import param
from ..io import PeriodicCallback
from ..layout import (
from ..util import fullpath
from ..viewable import Layoutable
from .base import CompositeWidget
from .button import Button
from .input import TextInput
from .select import CrossSelector

        Ensure that if unselecting a currently selected path and it
        is not in the current working directory then it is removed
        from the denylist.
        