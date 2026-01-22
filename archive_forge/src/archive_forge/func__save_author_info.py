from fastimport import helpers, processor
from ... import debug, errors, osutils, revision
from ...bzr import generate_ids, inventory, serializer
from ...trace import mutter, note, warning
from .helpers import mode_to_kind
def _save_author_info(self, rev_props):
    author = self.command.author
    if author is None:
        return
    if self.command.more_authors:
        authors = [author] + self.command.more_authors
        author_ids = [self._format_name_email('author', a[0], a[1]) for a in authors]
    elif author != self.command.committer:
        author_ids = [self._format_name_email('author', author[0], author[1])]
    else:
        return
    rev_props['authors'] = '\n'.join(author_ids)