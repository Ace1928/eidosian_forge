from io import BytesIO
from . import osutils, progress, trace
from .i18n import gettext
from .ui import ui_factory
Guess which files to rename, and perform the rename.

        We assume that unversioned files and missing files indicate that
        versioned files have been renamed outside of Bazaar.

        :param from_tree: A tree to compare from
        :param to_tree: A write-locked working tree.
        