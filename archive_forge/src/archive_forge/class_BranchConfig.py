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
class BranchConfig(Config):
    """A configuration object giving the policy for a branch."""

    def __init__(self, branch):
        super().__init__()
        self._location_config = None
        self._branch_data_config = None
        self._global_config = None
        self.branch = branch
        self.option_sources = (self._get_location_config, self._get_branch_data_config, self._get_global_config)

    def config_id(self):
        return 'branch'

    def _get_branch_data_config(self):
        if self._branch_data_config is None:
            self._branch_data_config = TreeConfig(self.branch)
            self._branch_data_config.config_id = self.config_id
        return self._branch_data_config

    def _get_location_config(self):
        if self._location_config is None:
            if self.branch.base is None:
                self.branch.base = 'memory://'
            self._location_config = LocationConfig(self.branch.base)
        return self._location_config

    def _get_global_config(self):
        if self._global_config is None:
            self._global_config = GlobalConfig()
        return self._global_config

    def _get_best_value(self, option_name):
        """This returns a user option from local, tree or global config.

        They are tried in that order.  Use get_safe_value if trusted values
        are necessary.
        """
        for source in self.option_sources:
            value = getattr(source(), option_name)()
            if value is not None:
                return value
        return None

    def _get_safe_value(self, option_name):
        """This variant of get_best_value never returns untrusted values.

        It does not return values from the branch data, because the branch may
        not be controlled by the user.

        We may wish to allow locations.conf to control whether branches are
        trusted in the future.
        """
        for source in (self._get_location_config, self._get_global_config):
            value = getattr(source(), option_name)()
            if value is not None:
                return value
        return None

    def _get_user_id(self):
        """Return the full user id for the branch.

        e.g. "John Hacker <jhacker@example.com>"
        This is looked up in the email controlfile for the branch.
        """
        return self._get_best_value('_get_user_id')

    def _get_change_editor(self):
        return self._get_best_value('_get_change_editor')

    def _get_signature_checking(self):
        """See Config._get_signature_checking."""
        return self._get_best_value('_get_signature_checking')

    def _get_signing_policy(self):
        """See Config._get_signing_policy."""
        return self._get_best_value('_get_signing_policy')

    def _get_user_option(self, option_name):
        """See Config._get_user_option."""
        for source in self.option_sources:
            value = source()._get_user_option(option_name)
            if value is not None:
                return value
        return None

    def _get_sections(self, name=None):
        """See IniBasedConfig.get_sections()."""
        for source in self.option_sources:
            yield from source()._get_sections(name)

    def _get_options(self, sections=None):
        yield from self._get_location_config()._get_options()
        branch_config = self._get_branch_data_config()
        if sections is None:
            sections = [('DEFAULT', branch_config._get_parser())]
        config_id = self.config_id()
        for section_name, section in sections:
            for name, value in section.iteritems():
                yield (name, value, section_name, config_id, branch_config._get_parser())
        yield from self._get_global_config()._get_options()

    def set_user_option(self, name, value, store=STORE_BRANCH, warn_masked=False):
        if store == STORE_BRANCH:
            self._get_branch_data_config().set_option(value, name)
        elif store == STORE_GLOBAL:
            self._get_global_config().set_user_option(name, value)
        else:
            self._get_location_config().set_user_option(name, value, store)
        if not warn_masked:
            return
        if store in (STORE_GLOBAL, STORE_BRANCH):
            mask_value = self._get_location_config().get_user_option(name)
            if mask_value is not None:
                trace.warning('Value "%s" is masked by "%s" from locations.conf', value, mask_value)
            elif store == STORE_GLOBAL:
                branch_config = self._get_branch_data_config()
                mask_value = branch_config.get_user_option(name)
                if mask_value is not None:
                    trace.warning('Value "%s" is masked by "%s" from branch.conf', value, mask_value)

    def remove_user_option(self, option_name, section_name=None):
        self._get_branch_data_config().remove_option(option_name, section_name)

    def _post_commit(self):
        """See Config.post_commit."""
        return self._get_safe_value('_post_commit')

    def _get_nickname(self):
        value = self._get_explicit_nickname()
        if value is not None:
            return value
        if self.branch.name:
            return self.branch.name
        return urlutils.unescape(self.branch.base.split('/')[-2])

    def has_explicit_nickname(self):
        """Return true if a nickname has been explicitly assigned."""
        return self._get_explicit_nickname() is not None

    def _get_explicit_nickname(self):
        return self._get_best_value('_get_nickname')

    def _log_format(self):
        """See Config.log_format."""
        return self._get_best_value('_log_format')

    def _validate_signatures_in_log(self):
        """See Config.validate_signatures_in_log."""
        return self._get_best_value('_validate_signatures_in_log')

    def _acceptable_keys(self):
        """See Config.acceptable_keys."""
        return self._get_best_value('_acceptable_keys')