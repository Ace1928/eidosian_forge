from __future__ import print_function
import collections
import re
import sys
import codecs
from . import (
from .helpers import (
class ImportParser(LineBasedParser):

    def __init__(self, input_stream, verbose=False, output=sys.stdout, user_mapper=None, strict=True):
        """A Parser of import commands.

        :param input_stream: the file-like object to read from
        :param verbose: display extra information of not
        :param output: the file-like object to write messages to (YAGNI?)
        :param user_mapper: if not None, the UserMapper used to adjust
          user-ids for authors, committers and taggers.
        :param strict: Raise errors on strictly invalid data
        """
        LineBasedParser.__init__(self, input_stream)
        self.verbose = verbose
        self.output = output
        self.user_mapper = user_mapper
        self.strict = strict
        self.date_parser = None
        self.features = {}

    def warning(self, msg):
        sys.stderr.write('warning line %d: %s\n' % (self.lineno, msg))

    def iter_commands(self):
        """Iterator returning ImportCommand objects."""
        while True:
            line = self.next_line()
            if line is None:
                if b'done' in self.features:
                    raise errors.PrematureEndOfStream(self.lineno)
                break
            elif len(line) == 0 or line.startswith(b'#'):
                continue
            elif line.startswith(b'commit '):
                yield self._parse_commit(line[len(b'commit '):])
            elif line.startswith(b'blob'):
                yield self._parse_blob()
            elif line.startswith(b'done'):
                break
            elif line.startswith(b'progress '):
                yield commands.ProgressCommand(line[len(b'progress '):])
            elif line.startswith(b'reset '):
                yield self._parse_reset(line[len(b'reset '):])
            elif line.startswith(b'tag '):
                yield self._parse_tag(line[len(b'tag '):])
            elif line.startswith(b'checkpoint'):
                yield commands.CheckpointCommand()
            elif line.startswith(b'feature'):
                yield self._parse_feature(line[len(b'feature '):])
            else:
                self.abort(errors.InvalidCommand, line)

    def iter_file_commands(self):
        """Iterator returning FileCommand objects.

        If an invalid file command is found, the line is silently
        pushed back and iteration ends.
        """
        while True:
            line = self.next_line()
            if line is None:
                break
            elif len(line) == 0 or line.startswith(b'#'):
                continue
            elif line.startswith(b'M '):
                yield self._parse_file_modify(line[2:])
            elif line.startswith(b'D '):
                path = self._path(line[2:])
                yield commands.FileDeleteCommand(path)
            elif line.startswith(b'R '):
                old, new = self._path_pair(line[2:])
                yield commands.FileRenameCommand(old, new)
            elif line.startswith(b'C '):
                src, dest = self._path_pair(line[2:])
                yield commands.FileCopyCommand(src, dest)
            elif line.startswith(b'deleteall'):
                yield commands.FileDeleteAllCommand()
            else:
                self.push_line(line)
                break

    def _parse_blob(self):
        """Parse a blob command."""
        lineno = self.lineno
        mark = self._get_mark_if_any()
        data = self._get_data(b'blob')
        return commands.BlobCommand(mark, data, lineno)

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

    def _parse_feature(self, info):
        """Parse a feature command."""
        parts = info.split(b'=', 1)
        name = parts[0]
        if len(parts) > 1:
            value = self._path(parts[1])
        else:
            value = None
        self.features[name] = value
        return commands.FeatureCommand(name, value, lineno=self.lineno)

    def _parse_file_modify(self, info):
        """Parse a filemodify command within a commit.

        :param info: a string in the format "mode dataref path"
          (where dataref might be the hard-coded literal 'inline').
        """
        params = info.split(b' ', 2)
        path = self._path(params[2])
        mode = self._mode(params[0])
        if params[1] == b'inline':
            dataref = None
            data = self._get_data(b'filemodify')
        else:
            dataref = params[1]
            data = None
        return commands.FileModifyCommand(path, mode, dataref, data)

    def _parse_reset(self, ref):
        """Parse a reset command."""
        from_ = self._get_from()
        return commands.ResetCommand(ref, from_)

    def _parse_tag(self, name):
        """Parse a tag command."""
        from_ = self._get_from(b'tag')
        tagger = self._get_user_info(b'tag', b'tagger', accept_just_who=True)
        message = self._get_data(b'tag', b'message')
        return commands.TagCommand(name, from_, tagger, message)

    def _get_mark_if_any(self):
        """Parse a mark section."""
        line = self.next_line()
        if line.startswith(b'mark :'):
            return line[len(b'mark :'):]
        else:
            self.push_line(line)
            return None

    def _get_from(self, required_for=None):
        """Parse a from section."""
        line = self.next_line()
        if line is None:
            return None
        elif line.startswith(b'from '):
            return line[len(b'from '):]
        elif required_for:
            self.abort(errors.MissingSection, required_for, 'from')
        else:
            self.push_line(line)
            return None

    def _get_merge(self):
        """Parse a merge section."""
        line = self.next_line()
        if line is None:
            return None
        elif line.startswith(b'merge '):
            return line[len(b'merge '):]
        else:
            self.push_line(line)
            return None

    def _get_property(self):
        """Parse a property section."""
        line = self.next_line()
        if line is None:
            return None
        elif line.startswith(b'property '):
            return self._name_value(line[len(b'property '):])
        else:
            self.push_line(line)
            return None

    def _get_user_info(self, cmd, section, required=True, accept_just_who=False):
        """Parse a user section."""
        line = self.next_line()
        if line.startswith(section + b' '):
            return self._who_when(line[len(section + b' '):], cmd, section, accept_just_who=accept_just_who)
        elif required:
            self.abort(errors.MissingSection, cmd, section)
        else:
            self.push_line(line)
            return None

    def _get_data(self, required_for, section=b'data'):
        """Parse a data section."""
        line = self.next_line()
        if line.startswith(b'data '):
            rest = line[len(b'data '):]
            if rest.startswith(b'<<'):
                return self.read_until(rest[2:])
            else:
                size = int(rest)
                read_bytes = self.read_bytes(size)
                next_line = self.input.readline()
                self.lineno += 1
                if len(next_line) > 1 or next_line != b'\n':
                    self.push_line(next_line[:-1])
                return read_bytes
        else:
            self.abort(errors.MissingSection, required_for, section)

    def _who_when(self, s, cmd, section, accept_just_who=False):
        """Parse who and when information from a string.

        :return: a tuple of (name,email,timestamp,timezone). name may be
            the empty string if only an email address was given.
        """
        match = _WHO_AND_WHEN_RE.search(s)
        if match:
            datestr = match.group(3).lstrip()
            if self.date_parser is None:
                if len(datestr.split(b' ')) == 2:
                    date_format = 'raw'
                elif datestr == b'now':
                    date_format = 'now'
                else:
                    date_format = 'rfc2822'
                self.date_parser = dates.DATE_PARSERS_BY_NAME[date_format]
            try:
                when = self.date_parser(datestr, self.lineno)
            except ValueError:
                print("failed to parse datestr '%s'" % (datestr,))
                raise
            name = match.group(1).rstrip()
            email = match.group(2)
        else:
            match = _WHO_RE.search(s)
            if accept_just_who and match:
                when = dates.DATE_PARSERS_BY_NAME['now']('now')
                name = match.group(1)
                email = match.group(2)
            elif self.strict:
                self.abort(errors.BadFormat, cmd, section, s)
            else:
                name = s
                email = None
                when = dates.DATE_PARSERS_BY_NAME['now']('now')
        if len(name) > 0:
            if name.endswith(b' '):
                name = name[:-1]
        if self.user_mapper:
            name, email = self.user_mapper.map_name_and_email(name, email)
        return Authorship(name, email, when[0], when[1])

    def _name_value(self, s):
        """Parse a (name,value) tuple from 'name value-length value'."""
        parts = s.split(b' ', 2)
        name = parts[0]
        if len(parts) == 1:
            value = None
        else:
            size = int(parts[1])
            value = parts[2]
            still_to_read = size - len(value)
            if still_to_read > 0:
                read_bytes = self.read_bytes(still_to_read)
                value += b'\n' + read_bytes[:still_to_read - 1]
        return (name, value)

    def _path(self, s):
        """Parse a path."""
        if s.startswith(b'"'):
            if not s.endswith(b'"'):
                self.abort(errors.BadFormat, '?', '?', s)
            else:
                return _unquote_c_string(s[1:-1])
        return s

    def _path_pair(self, s):
        """Parse two paths separated by a space."""
        if s.startswith(b'"'):
            parts = s[1:].split(b'" ', 1)
        else:
            parts = s.split(b' ', 1)
        if len(parts) != 2:
            self.abort(errors.BadFormat, '?', '?', s)
        elif parts[1].startswith(b'"') and parts[1].endswith(b'"'):
            parts[1] = parts[1][1:-1]
        elif parts[1].startswith(b'"') or parts[1].endswith(b'"'):
            self.abort(errors.BadFormat, '?', '?', s)
        return [_unquote_c_string(part) for part in parts]

    def _mode(self, s):
        """Check file mode format and parse into an int.

        :return: mode as integer
        """
        if s in [b'644', b'100644', b'0100644']:
            return 33188
        elif s in [b'755', b'100755', b'0100755']:
            return 33261
        elif s in [b'040000', b'0040000']:
            return 16384
        elif s in [b'120000', b'0120000']:
            return 40960
        elif s in [b'160000', b'0160000']:
            return 57344
        else:
            self.abort(errors.BadFormat, 'filemodify', 'mode', s)