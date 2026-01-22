from __future__ import absolute_import, division, print_function
import click
import os
import datetime
from typing import TYPE_CHECKING, Dict, Optional, Callable, Iterable
from incremental import Version
from incremental import Version
def _findPath(path, package):
    cwd = FilePath(path)
    src_dir = cwd.child('src').child(package.lower())
    current_dir = cwd.child(package.lower())
    if src_dir.isdir():
        return src_dir
    elif current_dir.isdir():
        return current_dir
    else:
        raise ValueError("Can't find under `./src` or `./`. Check the package name is right (note that we expect your package name to be lower cased), or pass it using '--path'.")