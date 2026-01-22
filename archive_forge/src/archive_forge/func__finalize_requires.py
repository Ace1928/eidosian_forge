import io
import itertools
import numbers
import os
import re
import sys
from contextlib import suppress
from glob import iglob
from pathlib import Path
from typing import List, Optional, Set
import distutils.cmd
import distutils.command
import distutils.core
import distutils.dist
import distutils.log
from distutils.debug import DEBUG
from distutils.errors import DistutilsOptionError, DistutilsSetupError
from distutils.fancy_getopt import translate_longopt
from distutils.util import strtobool
from .extern.more_itertools import partition, unique_everseen
from .extern.ordered_set import OrderedSet
from .extern.packaging.markers import InvalidMarker, Marker
from .extern.packaging.specifiers import InvalidSpecifier, SpecifierSet
from .extern.packaging.version import Version
from . import _entry_points
from . import _normalization
from . import _reqs
from . import command as _  # noqa  -- imported for side-effects
from ._importlib import metadata
from .config import setupcfg, pyprojecttoml
from .discovery import ConfigDiscovery
from .monkey import get_unpatched
from .warnings import InformationOnly, SetuptoolsDeprecationWarning
def _finalize_requires(self):
    """
        Set `metadata.python_requires` and fix environment markers
        in `install_requires` and `extras_require`.
        """
    if getattr(self, 'python_requires', None):
        self.metadata.python_requires = self.python_requires
    self._normalize_requires()
    self.metadata.install_requires = self.install_requires
    self.metadata.extras_require = self.extras_require
    if self.extras_require:
        for extra in self.extras_require.keys():
            extra = extra.split(':')[0]
            if extra:
                self.metadata.provides_extras.add(extra)