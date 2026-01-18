from __future__ import print_function, unicode_literals
import typing
import six
from .path import abspath, join, normpath, relpath
from .wrapfs import WrapFS
A version of `SubFS` which closes its parent when closed.