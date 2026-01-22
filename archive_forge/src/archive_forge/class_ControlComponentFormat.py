from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Type, cast
from .lazy_import import lazy_import
import textwrap
from breezy import (
from breezy.i18n import gettext
from . import errors, hooks, registry
from . import revision as _mod_revision
from . import trace
from . import transport as _mod_transport
class ControlComponentFormat:
    """A component that can live inside of a control directory."""
    upgrade_recommended = False

    def get_format_description(self):
        """Return the short description for this format."""
        raise NotImplementedError(self.get_format_description)

    def is_supported(self):
        """Is this format supported?

        Supported formats must be initializable and openable.
        Unsupported formats may not support initialization or committing or
        some other features depending on the reason for not being supported.
        """
        return True

    def check_support_status(self, allow_unsupported, recommend_upgrade=True, basedir=None):
        """Give an error or warning on old formats.

        Args:
          allow_unsupported: If true, allow opening
            formats that are strongly deprecated, and which may
            have limited functionality.

          recommend_upgrade: If true (default), warn
            the user through the ui object that they may wish
            to upgrade the object.
        """
        if not allow_unsupported and (not self.is_supported()):
            raise errors.UnsupportedFormatError(format=self)
        if recommend_upgrade and self.upgrade_recommended:
            ui.ui_factory.recommend_upgrade(self.get_format_description(), basedir)

    @classmethod
    def get_format_string(cls):
        raise NotImplementedError(cls.get_format_string)