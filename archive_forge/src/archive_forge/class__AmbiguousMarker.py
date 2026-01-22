import contextlib
import functools
import os
from collections import defaultdict
from functools import partial
from functools import wraps
from typing import (
from ..errors import FileError, OptionError
from ..extern.packaging.markers import default_environment as marker_env
from ..extern.packaging.requirements import InvalidRequirement, Requirement
from ..extern.packaging.specifiers import SpecifierSet
from ..extern.packaging.version import InvalidVersion, Version
from ..warnings import SetuptoolsDeprecationWarning
from . import expand
class _AmbiguousMarker(SetuptoolsDeprecationWarning):
    _SUMMARY = 'Ambiguous requirement marker.'
    _DETAILS = '\n    One of the parsed requirements in `{field}` looks like a valid environment marker:\n\n        {req!r}\n\n    Please make sure that the configuration file is correct.\n    You can use dangling lines to avoid this problem.\n    '
    _SEE_DOCS = 'userguide/declarative_config.html#opt-2'

    @classmethod
    def message(cls, **kw):
        docs = f'https://setuptools.pypa.io/en/latest/{cls._SEE_DOCS}'
        return cls._format(cls._SUMMARY, cls._DETAILS, see_url=docs, format_args=kw)