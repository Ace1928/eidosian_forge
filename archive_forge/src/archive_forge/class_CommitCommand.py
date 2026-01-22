from __future__ import division
import re
import stat
from .helpers import (
class CommitCommand(ImportCommand):

    def __init__(self, ref, mark, author, committer, message, from_, merges, file_iter, lineno=0, more_authors=None, properties=None):
        ImportCommand.__init__(self, b'commit')
        self.ref = ref
        self.mark = mark
        self.author = author
        self.committer = committer
        self.message = message
        self.from_ = from_
        self.merges = merges
        self.file_iter = file_iter
        self.more_authors = more_authors
        self.properties = properties
        self.lineno = lineno
        self._binary = [b'file_iter']
        if self.mark is None:
            self.id = b'@' + ('%d' % lineno).encode('utf-8')
        elif isinstance(self.mark, int):
            self.id = b':' + str(self.mark).encode('utf-8')
        else:
            self.id = b':' + self.mark

    def copy(self, **kwargs):
        if not isinstance(self.file_iter, list):
            self.file_iter = list(self.file_iter)
        fields = dict(((key, value) for key, value in self.__dict__.items() if key not in ('id', 'name') if not key.startswith('_')))
        fields.update(kwargs)
        return CommitCommand(**fields)

    def __bytes__(self):
        return self.to_string(include_file_contents=True)

    def to_string(self, use_features=True, include_file_contents=False):
        """
            @todo the name to_string is ambiguous since the method actually
                returns bytes.
        """
        if self.mark is None:
            mark_line = b''
        elif isinstance(self.mark, int):
            mark_line = b'\nmark :' + str(self.mark).encode('utf-8')
        else:
            mark_line = b'\nmark :' + self.mark
        if self.author is None:
            author_section = b''
        else:
            author_section = b'\nauthor ' + format_who_when(self.author)
            if use_features and self.more_authors:
                for author in self.more_authors:
                    author_section += b'\nauthor ' + format_who_when(author)
        committer = b'committer ' + format_who_when(self.committer)
        if self.message is None:
            msg_section = b''
        else:
            msg = self.message
            msg_section = ('\ndata %d\n' % len(msg)).encode('ascii') + msg
        if self.from_ is None:
            from_line = b''
        else:
            from_line = b'\nfrom ' + self.from_
        if self.merges is None:
            merge_lines = b''
        else:
            merge_lines = b''.join([b'\nmerge ' + m for m in self.merges])
        if use_features and self.properties:
            property_lines = []
            for name in sorted(self.properties):
                value = self.properties[name]
                property_lines.append(b'\n' + format_property(name, value))
            properties_section = b''.join(property_lines)
        else:
            properties_section = b''
        if self.file_iter is None:
            filecommands = b''
        elif include_file_contents:
            filecommands = b''.join([b'\n' + bytes(c) for c in self.iter_files()])
        else:
            filecommands = b''.join([b'\n' + str(c) for c in self.iter_files()])
        return b''.join([b'commit ', self.ref, mark_line, author_section + b'\n', committer, msg_section, from_line, merge_lines, properties_section, filecommands])

    def dump_str(self, names=None, child_lists=None, verbose=False):
        result = [ImportCommand.dump_str(self, names, verbose=verbose)]
        for f in self.iter_files():
            if child_lists is None:
                continue
            try:
                child_names = child_lists[f.name]
            except KeyError:
                continue
            result.append('\t%s' % f.dump_str(child_names, verbose=verbose))
        return '\n'.join(result)

    def iter_files(self):
        """Iterate over files."""
        if callable(self.file_iter):
            return self.file_iter()
        return iter(self.file_iter)