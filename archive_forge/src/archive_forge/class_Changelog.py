import email.utils
import logging
import os
import re
import socket
from debian.debian_support import Version
class Changelog(object):
    """Represents a debian/changelog file.

    To get the properly formatted changelog back out of the object
    merely call `str()` on it. The returned string should be a properly
    formatted changelog.

    :param file: str, list of str, or file-like.
        The contents of the changelog, either as a ``str``, ``unicode`` object,
        or an iterator of lines such as a filehandle, (each line is either a
        ``str`` or ``unicode``)
    :param max_blocks: int, optional (Default: ``None``, no limit)
        The maximum number of blocks to parse from the input.
    :param allow_empty_author: bool, optional (Default: `False`),
        Whether to allow an empty author in the trailer line of a change
        block.
    :param strict: bool, optional (Default: ``False``, use a warning)
        Whether to raise an exception if there are errors.
    :param encoding: str,
        If the input is a str or iterator of str, the encoding to use when
        interpreting the input.

    There are a number of errors that may be thrown by the module:

    - :class:`ChangelogParseError`:
      Indicates that the changelog could not be parsed, i.e. there is a line
      that does not conform to the requirements, or a line was found out of
      its normal position. May be thrown when using the method
      `parse_changelog`.
      The constructor will not throw this exception.
    - :class:`ChangelogCreateError`:
      Some information required to create the changelog was not available.
      This can be thrown when `str()` is used on the object, and will occur
      if a required value is `None`.
    - :class:`VersionError`:
      The string used to create a Version object cannot be parsed as it
      doesn't conform to the specification of a version number. Can be
      thrown when creating a Changelog object from an existing changelog,
      or instantiating a Version object directly to assign to the version
      attribute of a Changelog object.

    If you have a changelog that may have no author information yet as
    it is still a work in progress, i.e. the author line is just::

        --

    rather than::

        -- Author <author@debian.org>  Thu, 12 Dec 2006 12:23:34 +0000

    then you can pass ``allow_empty_author=True`` to the Changelog
    constructor. If you do this then the ``author`` and ``date``
    attributes may be ``None``.

    """

    def __init__(self, file=None, max_blocks=None, allow_empty_author=False, strict=False, encoding='utf-8'):
        self._encoding = encoding
        self._blocks = []
        self.initial_blank_lines = []
        if file is not None:
            self.parse_changelog(file, max_blocks=max_blocks, allow_empty_author=allow_empty_author, strict=strict)

    @staticmethod
    def _parse_error(message, strict):
        if strict:
            raise ChangelogParseError(message)
        logger.warning(message)

    def parse_changelog(self, file, max_blocks=None, allow_empty_author=False, strict=True, encoding=None):
        """ Read and parse a changelog file

        If you create an Changelog object without specifying a changelog
        file, you can parse a changelog file with this method. If the
        changelog doesn't parse cleanly, a :class:`ChangelogParseError`
        exception is thrown. The constructor will parse the changelog on
        a best effort basis.
        """
        first_heading = 'first heading'
        next_heading_or_eof = 'next heading of EOF'
        start_of_change_data = 'start of change data'
        more_changes_or_trailer = 'more change data or trailer'
        slurp_to_end = 'slurp to end'
        encoding = encoding or self._encoding
        if file is None:
            self._parse_error('Empty changelog file.', strict)
            return
        self._blocks = []
        self.initial_blank_lines = []
        current_block = ChangeBlock(encoding=encoding)
        changes = []
        state = first_heading
        old_state = None
        if isinstance(file, bytes):
            file = file.decode(encoding)
        if isinstance(file, str):
            if not file.strip():
                self._parse_error('Empty changelog file.', strict)
                return
            file = file.splitlines()
        for line in file:
            if not isinstance(line, str):
                line = line.decode(encoding)
            line = line.rstrip('\n')
            if state in (first_heading, next_heading_or_eof):
                top_match = topline.match(line)
                blank_match = blankline.match(line)
                if top_match is not None:
                    if max_blocks is not None and len(self._blocks) >= max_blocks:
                        return
                    current_block.package = top_match.group(1)
                    current_block._raw_version = top_match.group(2)
                    current_block.distributions = top_match.group(3).lstrip()
                    pairs = line.split(';', 1)[1]
                    all_keys = {}
                    other_pairs = {}
                    for pair in pairs.split(','):
                        pair = pair.strip()
                        kv_match = keyvalue.match(pair)
                        if kv_match is None:
                            self._parse_error("Invalid key-value pair after ';': %s" % pair, strict)
                            continue
                        key = kv_match.group(1)
                        value = kv_match.group(2)
                        if key.lower() in all_keys:
                            self._parse_error('Repeated key-value: %s' % key.lower(), strict)
                        all_keys[key.lower()] = value
                        if key.lower() == 'urgency':
                            val_match = value_re.match(value)
                            if val_match is None:
                                self._parse_error('Badly formatted urgency value: %s' % value, strict)
                            else:
                                current_block.urgency = val_match.group(1)
                                comment = val_match.group(2)
                                if comment is not None:
                                    current_block.urgency_comment = comment
                        else:
                            other_pairs[key] = value
                    current_block.other_pairs = other_pairs
                    state = start_of_change_data
                elif blank_match is not None:
                    if state == first_heading:
                        self.initial_blank_lines.append(line)
                    else:
                        self._blocks[-1].add_trailing_line(line)
                else:
                    emacs_match = emacs_variables.match(line)
                    vim_match = vim_variables.match(line)
                    cvs_match = cvs_keyword.match(line)
                    comments_match = comments.match(line)
                    more_comments_match = more_comments.match(line)
                    if (emacs_match is not None or vim_match is not None) and state != first_heading:
                        self._blocks[-1].add_trailing_line(line)
                        old_state = state
                        state = slurp_to_end
                        continue
                    if cvs_match is not None or comments_match is not None or more_comments_match is not None:
                        if state == first_heading:
                            self.initial_blank_lines.append(line)
                        else:
                            self._blocks[-1].add_trailing_line(line)
                        continue
                    if (old_format_re1.match(line) is not None or old_format_re2.match(line) is not None or old_format_re3.match(line) is not None or (old_format_re4.match(line) is not None) or (old_format_re5.match(line) is not None) or (old_format_re6.match(line) is not None) or (old_format_re7.match(line) is not None) or (old_format_re8.match(line) is not None)) and state != first_heading:
                        self._blocks[-1].add_trailing_line(line)
                        old_state = state
                        state = slurp_to_end
                        continue
                    self._parse_error('Unexpected line while looking for %s: %s' % (state, line), strict)
                    if state == first_heading:
                        self.initial_blank_lines.append(line)
                    else:
                        self._blocks[-1].add_trailing_line(line)
            elif state in (start_of_change_data, more_changes_or_trailer):
                change_match = changere.match(line)
                end_match = endline.match(line)
                end_no_details_match = endline_nodetails.match(line)
                blank_match = blankline.match(line)
                if change_match is not None:
                    changes.append(line)
                    state = more_changes_or_trailer
                elif end_match is not None:
                    if end_match.group(3) != '  ':
                        self._parse_error('Badly formatted trailer line: %s' % line, strict)
                        current_block._trailer_separator = end_match.group(3)
                    current_block.author = '%s <%s>' % (end_match.group(1), end_match.group(2))
                    current_block.date = end_match.group(4)
                    current_block._changes = changes
                    self._blocks.append(current_block)
                    changes = []
                    current_block = ChangeBlock(encoding=encoding)
                    state = next_heading_or_eof
                elif end_no_details_match is not None:
                    if not allow_empty_author:
                        self._parse_error('Badly formatted trailer line: %s' % line, strict)
                        continue
                    current_block._changes = changes
                    self._blocks.append(current_block)
                    changes = []
                    current_block = ChangeBlock(encoding=encoding)
                    state = next_heading_or_eof
                elif blank_match is not None:
                    changes.append(line)
                else:
                    cvs_match = cvs_keyword.match(line)
                    comments_match = comments.match(line)
                    more_comments_match = more_comments.match(line)
                    if cvs_match is not None or comments_match is not None or more_comments_match is not None:
                        changes.append(line)
                        continue
                    self._parse_error('Unexpected line while looking for %s: %s' % (state, line), strict)
                    changes.append(line)
            elif state == slurp_to_end:
                if old_state == next_heading_or_eof:
                    self._blocks[-1].add_trailing_line(line)
                else:
                    changes.append(line)
            else:
                assert False, 'Unknown state: %s' % state
        if state not in (next_heading_or_eof, slurp_to_end) or (state == slurp_to_end and old_state != next_heading_or_eof):
            self._parse_error('Found eof where expected %s' % state, strict)
            current_block._changes = changes
            current_block._no_trailer = True
            self._blocks.append(current_block)

    def get_version(self):
        """Return a Version object for the last version"""
        return self._blocks[0].version

    def set_version(self, version):
        """Set the version of the last changelog block

        version can be a full version string, or a Version object
        """
        self._blocks[0].version = Version(version)
    version = property(get_version, set_version, doc='Version object for latest changelog block.\n            (Property that can both get and set the version.)')
    full_version = property(lambda self: self.version.full_version, doc='The full version number of the last version')
    epoch = property(lambda self: self.version.epoch, doc='The epoch number of the last revision, or `None` if no epoch was used.')
    debian_version = property(lambda self: self.version.debian_revision, doc='The debian part of the version number of the last version.')
    debian_revision = property(lambda self: self.version.debian_revision, doc='The debian part of the version number of the last version.')
    upstream_version = property(lambda self: self.version.upstream_version, doc='The upstream part of the version number of the last version.')

    def get_package(self):
        """Returns the name of the package in the last entry."""
        return self._blocks[0].package

    def set_package(self, package):
        """ set the name of the package in the last entry. """
        self._blocks[0].package = package
    package = property(get_package, set_package, doc='Name of the package in the last version')

    def get_versions(self):
        return self.versions

    @property
    def versions(self):
        """Returns a list of :class:`debian.debian_support.Version` objects
        that are listed in the changelog."""
        return [block.version for block in self._blocks]

    def _raw_versions(self):
        return [block._raw_version for block in self._blocks]

    def _format(self, allow_missing_author=False):
        pieces = []
        for line in self.initial_blank_lines:
            pieces.append(line + '\n')
        for block in self._blocks:
            pieces.append(block._format(allow_missing_author=allow_missing_author))
        return ''.join(pieces)

    def __str__(self):
        return self._format()

    def __bytes__(self):
        return str(self).encode(self._encoding)

    def __iter__(self):
        return iter(self._blocks)

    def __getitem__(self, n):
        """ select a changelog entry by number, version string, or Version

        :param n: integer or str representing a version or Version object
        """
        if isinstance(n, str):
            return self[Version(n)]
        if isinstance(n, int):
            idx = n
        else:
            idx = self.versions.index(n)
        return self._blocks[idx]

    def __len__(self):
        return len(self._blocks)

    def set_distributions(self, distributions):
        self._blocks[0].distributions = distributions
    distributions = property(lambda self: self._blocks[0].distributions, set_distributions, doc='A string indicating the distributions that the package will be uploaded to\nin the most recent version.')

    def set_urgency(self, urgency):
        self._blocks[0].urgency = urgency
    urgency = property(lambda self: self._blocks[0].urgency, set_urgency, doc='A string indicating the urgency with which the most recent version will\nbe uploaded.')

    def add_change(self, change):
        """ and a new dot point to a changelog entry

        Adds a change entry to the most recent version. The change entry
        should conform to the required format of the changelog (i.e. start
        with two spaces). No line wrapping or anything will be performed,
        so it is advisable to do this yourself if it is a long entry. The
        change will be appended to the current changes, no support is
        provided for per-maintainer changes.
        """
        self._blocks[0].add_change(change)

    def set_author(self, author):
        """ set the author of the top changelog entry """
        self._blocks[0].author = author
    author = property(lambda self: self._blocks[0].author, set_author, doc='        The author of the most recent change.\n        This should be a properly formatted name/email pair.')

    def set_date(self, date):
        """ set the date of the top changelog entry

        :param date: str
            a properly formatted date string (`date -R` format; see Policy)
        """
        self._blocks[0].date = date
    date = property(lambda self: self._blocks[0].date, set_date, doc='        The date associated with the current entry.\n        Should be a properly formatted string with the date and timezone.\n        See the :func:`format_date()` function.')

    def new_block(self, package=None, version=None, distributions=None, urgency=None, urgency_comment=None, changes=None, author=None, date=None, other_pairs=None, encoding=None):
        """ Add a new changelog block to the changelog

        Start a new :class:`ChangeBlock` entry representing a new version
        of the package. The arguments (all optional) are passed directly
        to the :class:`ChangeBlock` constructor; they specify the values
        that can be provided to the `set_*` methods of this class. If
        they are omitted the associated attributes *must* be assigned to
        before the changelog is formatted as a str or written to a file.
        """
        encoding = encoding or self._encoding
        block = ChangeBlock(package, version, distributions, urgency, urgency_comment, changes, author, date, other_pairs, encoding)
        if self._blocks:
            block.add_trailing_line('')
        self._blocks.insert(0, block)

    def write_to_open_file(self, filehandle):
        """ Write the changelog entry to a filehandle

        Write the changelog out to the filehandle passed. The file argument
        must be an open file object.
        """
        filehandle.write(str(self))