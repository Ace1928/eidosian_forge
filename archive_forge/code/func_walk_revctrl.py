from distutils import log
import distutils.command.sdist as orig
import os
import sys
import io
import contextlib
from itertools import chain
from .._importlib import metadata
from .build import _ORIGINAL_SUBCOMMANDS
def walk_revctrl(dirname=''):
    """Find all files under revision control"""
    for ep in metadata.entry_points(group='setuptools.file_finders'):
        for item in ep.load()(dirname):
            yield item