import io
import os
import shlex
import sys
import tokenize
import shutil
import contextlib
import tempfile
import warnings
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union
import setuptools
import distutils
from . import errors
from ._path import same_path
from ._reqs import parse_strings
from .warnings import SetuptoolsDeprecationWarning
from distutils.util import strtobool
def _arbitrary_args(self, config_settings: _ConfigSettings) -> Iterator[str]:
    """
        Users may expect to pass arbitrary lists of arguments to a command
        via "--global-option" (example provided in PEP 517 of a "escape hatch").

        >>> fn = _ConfigSettingsTranslator()._arbitrary_args
        >>> list(fn(None))
        []
        >>> list(fn({}))
        []
        >>> list(fn({'--build-option': 'foo'}))
        ['foo']
        >>> list(fn({'--build-option': ['foo']}))
        ['foo']
        >>> list(fn({'--build-option': 'foo'}))
        ['foo']
        >>> list(fn({'--build-option': 'foo bar'}))
        ['foo', 'bar']
        >>> list(fn({'--global-option': 'foo'}))
        []
        """
    yield from self._get_config('--build-option', config_settings)