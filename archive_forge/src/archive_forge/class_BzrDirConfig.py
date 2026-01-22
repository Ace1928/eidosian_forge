import os
import sys
from io import BytesIO
from typing import Callable, Dict, Iterable, Tuple, cast
import configobj
import breezy
from .lazy_import import lazy_import
import errno
import fnmatch
import re
from breezy import (
from breezy.i18n import gettext
from . import (bedding, commands, errors, hooks, lazy_regex, registry, trace,
from .option import Option as CommandOption
class BzrDirConfig:

    def __init__(self, bzrdir):
        self._bzrdir = bzrdir
        self._config = bzrdir._get_config()

    def set_default_stack_on(self, value):
        """Set the default stacking location.

        It may be set to a location, or None.

        This policy affects all branches contained by this control dir, except
        for those under repositories.
        """
        if self._config is None:
            raise errors.BzrError('Cannot set configuration in %s' % self._bzrdir)
        if value is None:
            self._config.set_option('', 'default_stack_on')
        else:
            self._config.set_option(value, 'default_stack_on')

    def get_default_stack_on(self):
        """Return the default stacking location.

        This will either be a location, or None.

        This policy affects all branches contained by this control dir, except
        for those under repositories.
        """
        if self._config is None:
            return None
        value = self._config.get_option('default_stack_on')
        if value == '':
            value = None
        return value