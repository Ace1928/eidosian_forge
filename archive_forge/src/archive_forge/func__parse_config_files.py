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
def _parse_config_files(self, filenames=None):
    """
        Adapted from distutils.dist.Distribution.parse_config_files,
        this method provides the same functionality in subtly-improved
        ways.
        """
    from configparser import ConfigParser
    ignore_options = [] if sys.prefix == sys.base_prefix else ['install-base', 'install-platbase', 'install-lib', 'install-platlib', 'install-purelib', 'install-headers', 'install-scripts', 'install-data', 'prefix', 'exec-prefix', 'home', 'user', 'root']
    ignore_options = frozenset(ignore_options)
    if filenames is None:
        filenames = self.find_config_files()
    if DEBUG:
        self.announce('Distribution.parse_config_files():')
    parser = ConfigParser()
    parser.optionxform = str
    for filename in filenames:
        with open(filename, encoding='utf-8') as reader:
            if DEBUG:
                self.announce('  reading {filename}'.format(**locals()))
            parser.read_file(reader)
        for section in parser.sections():
            options = parser.options(section)
            opt_dict = self.get_option_dict(section)
            for opt in options:
                if opt == '__name__' or opt in ignore_options:
                    continue
                val = parser.get(section, opt)
                opt = self.warn_dash_deprecation(opt, section)
                opt = self.make_option_lowercase(opt, section)
                opt_dict[opt] = (filename, val)
        parser.__init__()
    if 'global' not in self.command_options:
        return
    for opt, (src, val) in self.command_options['global'].items():
        alias = self.negative_opt.get(opt)
        if alias:
            val = not strtobool(val)
        elif opt in ('verbose', 'dry_run'):
            val = strtobool(val)
        try:
            setattr(self, alias or opt, val)
        except ValueError as e:
            raise DistutilsOptionError(e) from e