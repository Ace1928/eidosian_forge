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
class TransportConfig:
    """A Config that reads/writes a config file on a Transport.

    It is a low-level object that considers config data to be name/value pairs
    that may be associated with a section.  Assigning meaning to these values
    is done at higher levels like TreeConfig.
    """

    def __init__(self, transport, filename):
        self._transport = transport
        self._filename = filename

    def get_option(self, name, section=None, default=None):
        """Return the value associated with a named option.

        Args:
          name: The name of the value
          section: The section the option is in (if any)
          default: The value to return if the value is not set
        Returns: The value or default value
        """
        configobj = self._get_configobj()
        if section is None:
            section_obj = configobj
        else:
            try:
                section_obj = configobj[section]
            except KeyError:
                return default
        value = section_obj.get(name, default)
        for hook in OldConfigHooks['get']:
            hook(self, name, value)
        return value

    def set_option(self, value, name, section=None):
        """Set the value associated with a named option.

        Args:
          value: The value to set
          name: The name of the value to set
          section: The section the option is in (if any)
        """
        configobj = self._get_configobj()
        if section is None:
            configobj[name] = value
        else:
            configobj.setdefault(section, {})[name] = value
        for hook in OldConfigHooks['set']:
            hook(self, name, value)
        self._set_configobj(configobj)

    def remove_option(self, option_name, section_name=None):
        configobj = self._get_configobj()
        if section_name is None:
            del configobj[option_name]
        else:
            del configobj[section_name][option_name]
        for hook in OldConfigHooks['remove']:
            hook(self, option_name)
        self._set_configobj(configobj)

    def _get_config_file(self):
        try:
            f = BytesIO(self._transport.get_bytes(self._filename))
            for hook in OldConfigHooks['load']:
                hook(self)
            return f
        except transport.NoSuchFile:
            return BytesIO()
        except errors.PermissionDenied:
            trace.warning('Permission denied while trying to open configuration file %s.', urlutils.unescape_for_display(urlutils.join(self._transport.base, self._filename), 'utf-8'))
            return BytesIO()

    def _external_url(self):
        return urlutils.join(self._transport.external_url(), self._filename)

    def _get_configobj(self):
        f = self._get_config_file()
        try:
            try:
                conf = ConfigObj(f, encoding='utf-8')
            except configobj.ConfigObjError as e:
                raise ParseConfigError(e.errors, self._external_url())
            except UnicodeDecodeError:
                raise ConfigContentError(self._external_url())
        finally:
            f.close()
        return conf

    def _set_configobj(self, configobj):
        out_file = BytesIO()
        configobj.write(out_file)
        out_file.seek(0)
        self._transport.put_file(self._filename, out_file)
        for hook in OldConfigHooks['save']:
            hook(self)