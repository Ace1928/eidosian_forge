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
class IniBasedConfig(Config):
    """A configuration policy that draws from ini files."""

    def __init__(self, file_name=None):
        """Base class for configuration files using an ini-like syntax.

        Args:
          file_name: The configuration file path.
        """
        super().__init__()
        self.file_name = file_name
        self.file_name = file_name
        self._content = None
        self._parser = None

    @classmethod
    def from_string(cls, str_or_unicode, file_name=None, save=False):
        """Create a config object from a string.

        Args:
          str_or_unicode: A string representing the file content. This
            will be utf-8 encoded.
          file_name: The configuration file path.
          _save: Whether the file should be saved upon creation.
        """
        conf = cls(file_name=file_name)
        conf._create_from_string(str_or_unicode, save)
        return conf

    def _create_from_string(self, str_or_unicode, save):
        if isinstance(str_or_unicode, str):
            str_or_unicode = str_or_unicode.encode('utf-8')
        self._content = BytesIO(str_or_unicode)
        if save:
            self._write_config_file()

    def _get_parser(self):
        if self._parser is not None:
            return self._parser
        if self._content is not None:
            co_input = self._content
        elif self.file_name is None:
            raise AssertionError('We have no content to create the config')
        else:
            co_input = self.file_name
        try:
            self._parser = ConfigObj(co_input, encoding='utf-8')
        except configobj.ConfigObjError as e:
            raise ParseConfigError(e.errors, e.config.filename)
        except UnicodeDecodeError:
            raise ConfigContentError(self.file_name)
        self._parser.filename = self.file_name
        for hook in OldConfigHooks['load']:
            hook(self)
        return self._parser

    def reload(self):
        """Reload the config file from disk."""
        if self.file_name is None:
            raise AssertionError('We need a file name to reload the config')
        if self._parser is not None:
            self._parser.reload()
        for hook in ConfigHooks['load']:
            hook(self)

    def _get_matching_sections(self):
        """Return an ordered list of (section_name, extra_path) pairs.

        If the section contains inherited configuration, extra_path is
        a string containing the additional path components.
        """
        section = self._get_section()
        if section is not None:
            return [(section, '')]
        else:
            return []

    def _get_section(self):
        """Override this to define the section used by the config."""
        return 'DEFAULT'

    def _get_sections(self, name=None):
        """Returns an iterator of the sections specified by ``name``.

        Args:
          name: The section name. If None is supplied, the default
            configurations are yielded.

        Returns:
          A tuple (name, section, config_id) for all sections that will
          be walked by user_get_option() in the 'right' order. The first one
          is where set_user_option() will update the value.
        """
        parser = self._get_parser()
        if name is not None:
            yield (name, parser[name], self.config_id())
        else:
            yield (None, parser, self.config_id())

    def _get_options(self, sections=None):
        """Return an ordered list of (name, value, section, config_id) tuples.

        All options are returned with their associated value and the section
        they appeared in. ``config_id`` is a unique identifier for the
        configuration file the option is defined in.

        Args:
          sections: Default to ``_get_matching_sections`` if not
             specified. This gives a better control to daughter classes about
             which sections should be searched. This is a list of (name,
             configobj) tuples.
        """
        if sections is None:
            parser = self._get_parser()
            sections = []
            for section_name, _ in self._get_matching_sections():
                try:
                    section = parser[section_name]
                except KeyError:
                    continue
                sections.append((section_name, section))
        config_id = self.config_id()
        for section_name, section in sections:
            for name, value in section.iteritems():
                yield (name, parser._quote(value), section_name, config_id, parser)

    def _get_option_policy(self, section, option_name):
        """Return the policy for the given (section, option_name) pair."""
        return POLICY_NONE

    def _get_change_editor(self):
        return self.get_user_option('change_editor', expand=False)

    def _get_signature_checking(self):
        """See Config._get_signature_checking."""
        policy = self._get_user_option('check_signatures')
        if policy:
            return signature_policy_from_unicode(policy)

    def _get_signing_policy(self):
        """See Config._get_signing_policy"""
        policy = self._get_user_option('create_signatures')
        if policy:
            return signing_policy_from_unicode(policy)

    def _get_user_id(self):
        """Get the user id from the 'email' key in the current section."""
        return self._get_user_option('email')

    def _get_user_option(self, option_name):
        """See Config._get_user_option."""
        for section, extra_path in self._get_matching_sections():
            try:
                value = self._get_parser().get_value(section, option_name)
            except KeyError:
                continue
            policy = self._get_option_policy(section, option_name)
            if policy == POLICY_NONE:
                return value
            elif policy == POLICY_NORECURSE:
                if extra_path:
                    continue
                else:
                    return value
            elif policy == POLICY_APPENDPATH:
                if extra_path:
                    value = urlutils.join(value, extra_path)
                return value
            else:
                raise AssertionError('Unexpected config policy %r' % policy)
        else:
            return None

    def _log_format(self):
        """See Config.log_format."""
        return self._get_user_option('log_format')

    def _validate_signatures_in_log(self):
        """See Config.validate_signatures_in_log."""
        return self._get_user_option('validate_signatures_in_log')

    def _acceptable_keys(self):
        """See Config.acceptable_keys."""
        return self._get_user_option('acceptable_keys')

    def _post_commit(self):
        """See Config.post_commit."""
        return self._get_user_option('post_commit')

    def _get_alias(self, value):
        try:
            return self._get_parser().get_value('ALIASES', value)
        except KeyError:
            pass

    def _get_nickname(self):
        return self.get_user_option('nickname')

    def remove_user_option(self, option_name, section_name=None):
        """Remove a user option and save the configuration file.

        Args:
          option_name: The option to be removed.
          section_name: The section the option is defined in, default to
            the default section.
        """
        self.reload()
        parser = self._get_parser()
        if section_name is None:
            section = parser
        else:
            section = parser[section_name]
        try:
            del section[option_name]
        except KeyError:
            raise NoSuchConfigOption(option_name)
        self._write_config_file()
        for hook in OldConfigHooks['remove']:
            hook(self, option_name)

    def _write_config_file(self):
        if self.file_name is None:
            raise AssertionError('We cannot save, self.file_name is None')
        from . import atomicfile
        conf_dir = os.path.dirname(self.file_name)
        bedding.ensure_config_dir_exists(conf_dir)
        with atomicfile.AtomicFile(self.file_name) as atomic_file:
            self._get_parser().write(atomic_file)
        osutils.copy_ownership_from_path(self.file_name)
        for hook in OldConfigHooks['save']:
            hook(self)