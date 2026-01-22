from __future__ import print_function, unicode_literals
import typing
import six
from ._typing import Text
@property
def appending(self):
    """`bool`: `True` if the mode permits appending."""
    return 'a' in self