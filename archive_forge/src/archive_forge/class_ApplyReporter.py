import contextlib
import shutil
import sys
import tempfile
from io import BytesIO
import patiencediff
from . import (builtins, delta, diff, errors, osutils, patches, shelf,
from .i18n import gettext
class ApplyReporter(ShelfReporter):
    vocab = {'add file': gettext('Delete file "%(path)s"?'), 'binary': gettext('Apply binary changes?'), 'change kind': gettext('Change "%(path)s" from %(this)s to %(other)s?'), 'delete file': gettext('Add file "%(path)s"?'), 'final': gettext('Apply %d change(s)?'), 'hunk': gettext('Apply change?'), 'modify target': gettext('Change target of "%(path)s" from "%(this)s" to "%(other)s"?'), 'rename': gettext('Rename "%(this)s" => "%(other)s"?')}
    invert_diff = True

    def changes_destroyed(self):
        pass