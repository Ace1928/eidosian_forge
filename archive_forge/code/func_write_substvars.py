import contextlib
import errno
import re
import sys
import typing
from abc import ABC
from collections import OrderedDict
from collections.abc import MutableMapping
from types import TracebackType
from typing import Dict, Set, Optional, Union, Iterator, IO, Iterable, TYPE_CHECKING, Type
def write_substvars(self, fileobj):
    """Write a copy of the substvars to an open text file

        :param fileobj: The open file (should open in text mode using the UTF-8 encoding)
        """
    fileobj.writelines(('{}{}{}\n'.format(k, v.assignment_operator, v.resolve()) for k, v in self._vars.items()))