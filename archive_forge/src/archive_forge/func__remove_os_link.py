from distutils import log
import distutils.command.sdist as orig
import os
import sys
import io
import contextlib
from itertools import chain
from .._importlib import metadata
from .build import _ORIGINAL_SUBCOMMANDS
@staticmethod
@contextlib.contextmanager
def _remove_os_link():
    """
        In a context, remove and restore os.link if it exists
        """

    class NoValue:
        pass
    orig_val = getattr(os, 'link', NoValue)
    try:
        del os.link
    except Exception:
        pass
    try:
        yield
    finally:
        if orig_val is not NoValue:
            setattr(os, 'link', orig_val)