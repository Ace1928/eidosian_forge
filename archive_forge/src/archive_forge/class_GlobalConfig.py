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
class GlobalConfig(LockableConfig):
    """The configuration that should be used for a specific location."""

    def __init__(self):
        super().__init__(file_name=bedding.config_path())

    def config_id(self):
        return 'breezy'

    @classmethod
    def from_string(cls, str_or_unicode, save=False):
        """Create a config object from a string.

        Args:
          str_or_unicode: A string representing the file content. This
            will be utf-8 encoded.
          save: Whether the file should be saved upon creation.
        """
        conf = cls()
        conf._create_from_string(str_or_unicode, save)
        return conf

    def set_user_option(self, option, value):
        """Save option and its value in the configuration."""
        with self.lock_write():
            self._set_option(option, value, 'DEFAULT')

    def get_aliases(self):
        """Return the aliases section."""
        if 'ALIASES' in self._get_parser():
            return self._get_parser()['ALIASES']
        else:
            return {}

    def set_alias(self, alias_name, alias_command):
        """Save the alias in the configuration."""
        with self.lock_write():
            self._set_option(alias_name, alias_command, 'ALIASES')

    def unset_alias(self, alias_name):
        """Unset an existing alias."""
        with self.lock_write():
            self.reload()
            aliases = self._get_parser().get('ALIASES')
            if not aliases or alias_name not in aliases:
                raise NoSuchAlias(alias_name)
            del aliases[alias_name]
            self._write_config_file()

    def _set_option(self, option, value, section):
        self.reload()
        self._get_parser().setdefault(section, {})[option] = value
        self._write_config_file()
        for hook in OldConfigHooks['set']:
            hook(self, option, value)

    def _get_sections(self, name=None):
        """See IniBasedConfig._get_sections()."""
        parser = self._get_parser()
        if name in (None, 'DEFAULT'):
            name = 'DEFAULT'
            if 'DEFAULT' not in parser:
                parser['DEFAULT'] = {}
        yield (name, parser[name], self.config_id())

    def remove_user_option(self, option_name, section_name=None):
        if section_name is None:
            section_name = 'DEFAULT'
        with self.lock_write():
            super(LockableConfig, self).remove_user_option(option_name, section_name)