import contextlib
import sys
from typing import TYPE_CHECKING, Set, cast
from ..lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from .. import config, controldir, errors, lockdir
from .. import transport as _mod_transport
from ..trace import mutter, note, warning
from ..transport import do_catching_redirections, local
class ConvertMetaToColo(controldir.Converter):
    """Add colocated branch support."""

    def __init__(self, target_format):
        """Create a converter.that upgrades a metadir to the colo format.

        :param target_format: The final metadir format that is desired.
        """
        self.target_format = target_format

    def convert(self, to_convert, pb):
        """See Converter.convert()."""
        to_convert.transport.put_bytes('branch-format', self.target_format.as_string())
        return BzrDir.open_from_transport(to_convert.root_transport)