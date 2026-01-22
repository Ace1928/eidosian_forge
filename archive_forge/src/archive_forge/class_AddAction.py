import os
import sys
from . import errors, osutils, ui
from .i18n import gettext
class AddAction:
    """A class which defines what action to take when adding a file."""

    def __init__(self, to_file=None, should_print=None):
        """Initialize an action which prints added files to an output stream.

        :param to_file: The stream to write into. This is expected to take
            Unicode paths. If not supplied, it will default to ``sys.stdout``.
        :param should_print: If False, printing will be suppressed.
        """
        self._to_file = to_file
        if to_file is None:
            self._to_file = sys.stdout
        self.should_print = False
        if should_print is not None:
            self.should_print = should_print

    def __call__(self, inv, parent_ie, path, kind, _quote=osutils.quotefn):
        """Add path to inventory.

        The default action does nothing.

        :param inv: The inventory we are working with.
        :param path: The FastPath being added
        :param kind: The kind of the object being added.
        """
        if self.should_print:
            self._to_file.write('adding %s\n' % _quote(path))
        return None

    def skip_file(self, tree, path, kind, stat_value=None):
        """Test whether the given file should be skipped or not.

        The default action never skips. Note this is only called during
        recursive adds

        :param tree: The tree we are working in
        :param path: The path being added
        :param kind: The kind of object being added.
        :param stat: Stat result for this file, if available already
        :return bool. True if the file should be skipped (not added)
        """
        return False