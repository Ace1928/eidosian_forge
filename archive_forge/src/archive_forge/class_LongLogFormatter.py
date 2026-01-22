import codecs
import itertools
import re
import sys
from io import BytesIO
from typing import Callable, Dict, List
from warnings import warn
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext, ngettext
from . import errors, registry
from . import revision as _mod_revision
from . import revisionspec, trace
from . import transport as _mod_transport
from .osutils import (format_date,
from .tree import InterTree, find_previous_path
class LongLogFormatter(LogFormatter):
    supports_merge_revisions = True
    preferred_levels = 1
    supports_delta = True
    supports_tags = True
    supports_diff = True
    supports_signatures = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.show_timezone == 'original':
            self.date_string = self._date_string_original_timezone
        else:
            self.date_string = self._date_string_with_timezone

    def _date_string_with_timezone(self, rev):
        return format_date(rev.timestamp, rev.timezone or 0, self.show_timezone)

    def _date_string_original_timezone(self, rev):
        return format_date_with_offset_in_original_timezone(rev.timestamp, rev.timezone or 0)

    def log_revision(self, revision):
        """Log a revision, either merged or not."""
        indent = '    ' * revision.merge_depth
        lines = [_LONG_SEP]
        if revision.revno is not None:
            lines.append('revno: {}{}'.format(revision.revno, self.merge_marker(revision)))
        if revision.tags:
            lines.append('tags: %s' % ', '.join(sorted(revision.tags)))
        if self.show_ids or revision.revno is None:
            lines.append('revision-id: %s' % (revision.rev.revision_id.decode('utf-8'),))
        if self.show_ids:
            for parent_id in revision.rev.parent_ids:
                lines.append('parent: {}'.format(parent_id.decode('utf-8')))
        lines.extend(self.custom_properties(revision.rev))
        committer = revision.rev.committer
        authors = self.authors(revision.rev, 'all')
        if authors != [committer]:
            lines.append('author: {}'.format(', '.join(authors)))
        lines.append('committer: {}'.format(committer))
        branch_nick = revision.rev.properties.get('branch-nick', None)
        if branch_nick is not None:
            lines.append('branch nick: {}'.format(branch_nick))
        lines.append('timestamp: {}'.format(self.date_string(revision.rev)))
        if revision.signature is not None:
            lines.append('signature: ' + revision.signature)
        lines.append('message:')
        if not revision.rev.message:
            lines.append('  (no message)')
        else:
            message = revision.rev.message.rstrip('\r\n')
            for l in message.split('\n'):
                lines.append('  {}'.format(l))
        to_file = self.to_file
        to_file.write('{}{}\n'.format(indent, ('\n' + indent).join(lines)))
        if revision.delta is not None:
            from breezy.delta import report_delta
            report_delta(to_file, revision.delta, short_status=False, show_ids=self.show_ids, indent=indent)
        if revision.diff is not None:
            to_file.write(indent + 'diff:\n')
            to_file.flush()
            self.show_diff(self.to_exact_file, revision.diff, indent)
            self.to_exact_file.flush()

    def get_advice_separator(self):
        """Get the text separating the log from the closing advice."""
        return '-' * 60 + '\n'