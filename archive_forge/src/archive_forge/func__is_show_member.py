import inspect
import textwrap
import re
import pydoc
from warnings import warn
from collections import namedtuple
from collections.abc import Callable, Mapping
import copy
import sys
def _is_show_member(self, name):
    if self.show_inherited_members:
        return True
    if name not in self._cls.__dict__:
        return False
    return True