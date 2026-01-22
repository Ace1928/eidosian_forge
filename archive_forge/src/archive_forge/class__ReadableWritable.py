from __future__ import absolute_import, division, print_function
import click
import os
import datetime
from typing import TYPE_CHECKING, Dict, Optional, Callable, Iterable
from incremental import Version
from incremental import Version
class _ReadableWritable(Protocol):

    def read(self):
        pass

    def write(self, v):
        pass

    def __enter__(self):
        pass

    def __exit__(self, *args, **kwargs):
        pass