import contextlib
import shutil
import sys
import tempfile
from io import BytesIO
import patiencediff
from . import (builtins, delta, diff, errors, osutils, patches, shelf,
from .i18n import gettext
def write_diff(self, merger):
    """Write this operation's diff to self.write_diff_to."""
    tree_merger = merger.make_merger()
    tt = tree_merger.make_preview_transform()
    new_tree = tt.get_preview_tree()
    if self.write_diff_to is None:
        self.write_diff_to = ui.ui_factory.make_output_stream(encoding_type='exact')
    path_encoding = osutils.get_diff_header_encoding()
    diff.show_diff_trees(merger.this_tree, new_tree, self.write_diff_to, path_encoding=path_encoding)
    tt.finalize()