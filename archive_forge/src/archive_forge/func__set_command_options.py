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
def _set_command_options(self, command_obj, option_dict=None):
    """
        Set the options for 'command_obj' from 'option_dict'.  Basically
        this means copying elements of a dictionary ('option_dict') to
        attributes of an instance ('command').

        'command_obj' must be a Command instance.  If 'option_dict' is not
        supplied, uses the standard option dictionary for this command
        (from 'self.command_options').

        (Adopted from distutils.dist.Distribution._set_command_options)
        """
    command_name = command_obj.get_command_name()
    if option_dict is None:
        option_dict = self.get_option_dict(command_name)
    if DEBUG:
        self.announce("  setting options for '%s' command:" % command_name)
    for option, (source, value) in option_dict.items():
        if DEBUG:
            self.announce('    %s = %s (from %s)' % (option, value, source))
        try:
            bool_opts = [translate_longopt(o) for o in command_obj.boolean_options]
        except AttributeError:
            bool_opts = []
        try:
            neg_opt = command_obj.negative_opt
        except AttributeError:
            neg_opt = {}
        try:
            is_string = isinstance(value, str)
            if option in neg_opt and is_string:
                setattr(command_obj, neg_opt[option], not strtobool(value))
            elif option in bool_opts and is_string:
                setattr(command_obj, option, strtobool(value))
            elif hasattr(command_obj, option):
                setattr(command_obj, option, value)
            else:
                raise DistutilsOptionError("error in %s: command '%s' has no such option '%s'" % (source, command_name, option))
        except ValueError as e:
            raise DistutilsOptionError(e) from e