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
def _parse_attr(self, value, package_dir, root_dir: _Path):
    """Represents value as a module attribute.

        Examples:
            attr: package.attr
            attr: package.module.attr

        :param str value:
        :rtype: str
        """
    attr_directive = 'attr:'
    if not value.startswith(attr_directive):
        return value
    attr_desc = value.replace(attr_directive, '')
    package_dir.update(self.ensure_discovered.package_dir)
    return expand.read_attr(attr_desc, package_dir, root_dir)