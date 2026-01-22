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
class OptionRegistry(registry.Registry):
    """Register config options by their name.

    This overrides ``registry.Registry`` to simplify registration by acquiring
    some information from the option object itself.
    """

    def _check_option_name(self, option_name):
        """Ensures an option name is valid.

        Args:
          option_name: The name to validate.
        """
        if _option_ref_re.match('{%s}' % option_name) is None:
            raise IllegalOptionName(option_name)

    def register(self, option):
        """Register a new option to its name.

        Args:
          option: The option to register. Its name is used as the key.
        """
        self._check_option_name(option.name)
        super().register(option.name, option, help=option.help)

    def register_lazy(self, key, module_name, member_name):
        """Register a new option to be loaded on request.

        Args:
          key: the key to request the option later. Since the registration
            is lazy, it should be provided and match the option name.

          module_name: the python path to the module. Such as 'os.path'.

          member_name: the member of the module to return.  If empty or
                None, get() will return the module itself.
        """
        self._check_option_name(key)
        super().register_lazy(key, module_name, member_name)

    def get_help(self, key=None):
        """Get the help text associated with the given key"""
        option = self.get(key)
        the_help = option.help
        if callable(the_help):
            return the_help(self, key)
        return the_help