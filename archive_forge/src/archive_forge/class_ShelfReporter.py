import contextlib
import shutil
import sys
import tempfile
from io import BytesIO
import patiencediff
from . import (builtins, delta, diff, errors, osutils, patches, shelf,
from .i18n import gettext
class ShelfReporter:
    vocab = {'add file': gettext('Shelve adding file "%(path)s"?'), 'binary': gettext('Shelve binary changes?'), 'change kind': gettext('Shelve changing "%s" from %(other)s to %(this)s?'), 'delete file': gettext('Shelve removing file "%(path)s"?'), 'final': gettext('Shelve %d change(s)?'), 'hunk': gettext('Shelve?'), 'modify target': gettext('Shelve changing target of "%(path)s" from "%(other)s" to "%(this)s"?'), 'rename': gettext('Shelve renaming "%(other)s" => "%(this)s"?')}
    invert_diff = False

    def __init__(self):
        self.delta_reporter = delta._ChangeReporter()

    def no_changes(self):
        """Report that no changes were selected to apply."""
        trace.warning('No changes to shelve.')

    def shelved_id(self, shelf_id):
        """Report the id changes were shelved to."""
        trace.note(gettext('Changes shelved with id "%d".') % shelf_id)

    def changes_destroyed(self):
        """Report that changes were made without shelving."""
        trace.note(gettext('Selected changes destroyed.'))

    def selected_changes(self, transform):
        """Report the changes that were selected."""
        trace.note(gettext('Selected changes:'))
        changes = transform.iter_changes()
        delta.report_changes(changes, self.delta_reporter)

    def prompt_change(self, change):
        """Determine the prompt for a change to apply."""
        if change[0] == 'rename':
            vals = {'this': change[3], 'other': change[2]}
        elif change[0] == 'change kind':
            vals = {'path': change[4], 'other': change[2], 'this': change[3]}
        elif change[0] == 'modify target':
            vals = {'path': change[2], 'other': change[3], 'this': change[4]}
        else:
            vals = {'path': change[3]}
        prompt = self.vocab[change[0]] % vals
        return prompt