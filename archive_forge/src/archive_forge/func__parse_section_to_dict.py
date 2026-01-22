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
@classmethod
def _parse_section_to_dict(cls, section_options, values_parser=None):
    """Parses section options into a dictionary.

        Optionally applies a given parser to each value.

        :param dict section_options:
        :param callable values_parser: function with 1 arg corresponding to option value
        :rtype: dict
        """
    parser = (lambda _, v: values_parser(v)) if values_parser else lambda _, v: v
    return cls._parse_section_to_dict_with_key(section_options, parser)