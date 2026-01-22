from __future__ import print_function
import collections
import re
import sys
import codecs
from . import (
from .helpers import (
def _parse_commit(self, ref):
    """Parse a commit command."""
    lineno = self.lineno
    mark = self._get_mark_if_any()
    author = self._get_user_info(b'commit', b'author', False)
    more_authors = []
    while True:
        another_author = self._get_user_info(b'commit', b'author', False)
        if another_author is not None:
            more_authors.append(another_author)
        else:
            break
    committer = self._get_user_info(b'commit', b'committer')
    message = self._get_data(b'commit', b'message')
    from_ = self._get_from()
    merges = []
    while True:
        merge = self._get_merge()
        if merge is not None:
            these_merges = merge.split(b' ')
            merges.extend(these_merges)
        else:
            break
    properties = {}
    while True:
        name_value = self._get_property()
        if name_value is not None:
            name, value = name_value
            properties[name] = value
        else:
            break
    return commands.CommitCommand(ref, mark, author, committer, message, from_, merges, list(self.iter_file_commands()), lineno=lineno, more_authors=more_authors, properties=properties)