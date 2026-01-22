import email.utils
import logging
import os
import re
import socket
from debian.debian_support import Version
class ChangeBlock(object):
    """Holds all the information about one block from the changelog.

    See `deb-changelog(5)
    <https://manpages.debian.org/stretch/dpkg-dev/deb-changelog.5.html>`_
    for more details about the format of the changelog block and the
    necessary data.

    :param package: str, name of the package
    :param version: str or Version, version of the package
    :param distributions: str, distributions to which the package is
        released
    :param urgency: str, urgency of the upload
    :param urgency_comment: str, comment about the urgency setting
    :param changes: list of str, individual changelog entries for this
        block
    :param author: str, name and email address of the changelog author
    :param date: str, date of the changelog in RFC822 (`date -R`) format
    :param other_pairs: dict, key=value pairs from the header of the
        changelog, other than the urgency value that is specified
        separately
    :param encoding: specify the encoding to be used; note that Debian
        Policy mandates the use of UTF-8.
    """

    def __init__(self, package=None, version=None, distributions=None, urgency=None, urgency_comment=None, changes=None, author=None, date=None, other_pairs=None, encoding='utf-8'):
        self._raw_version = None
        self._set_version(version)
        self.package = package
        self.distributions = distributions
        self.urgency = urgency or 'unknown'
        self.urgency_comment = urgency_comment or ''
        self._changes = changes or []
        self.author = author
        self.date = date
        self._trailing = []
        self.other_pairs = other_pairs or {}
        self._encoding = encoding
        self._no_trailer = False
        self._trailer_separator = '  '

    def _get_version(self):
        if self._raw_version is None:
            return None
        return Version(self._raw_version)

    def _set_version(self, version):
        if version is not None:
            self._raw_version = str(version)
        else:
            self._raw_version = None
    version = property(_get_version, _set_version, doc='The package version that this block pertains to')

    def other_keys_normalised(self):
        """ Obtain a dict from the block header (other than urgency) """
        norm_dict = {}
        for key, value in self.other_pairs.items():
            key = key[0].upper() + key[1:].lower()
            m = xbcs_re.match(key)
            if m is None:
                key = 'XS-%s' % key
            norm_dict[key] = value
        return norm_dict

    def changes(self):
        """ Get the changelog entries for this block as a list of str """
        return self._changes

    def add_trailing_line(self, line):
        """ Add a sign-off (trailer) line to the block """
        self._trailing.append(line)

    def add_change(self, change):
        """ Append a change entry to the block """
        if not self._changes:
            self._changes = [change]
        else:
            changes = self._changes
            changes.reverse()
            added = False
            for i, ch_entry in enumerate(changes):
                m = blankline.match(ch_entry)
                if m is None:
                    changes.insert(i, change)
                    added = True
                    break
            changes.reverse()
            if not added:
                changes.append(change)
            self._changes = changes

    def _get_bugs_closed_generic(self, type_re):
        changes = ' '.join(self._changes)
        bugs = []
        for match in type_re.finditer(changes):
            closes_list = match.group(0)
            for bugmatch in re.finditer('\\d+', closes_list):
                bugs.append(int(bugmatch.group(0)))
        return bugs

    @property
    def bugs_closed(self):
        """ List of (Debian) bugs closed by the block """
        return self._get_bugs_closed_generic(closes)

    @property
    def lp_bugs_closed(self):
        """ List of Launchpad bugs closed by the block """
        return self._get_bugs_closed_generic(closeslp)

    def _format(self, allow_missing_author=False):
        block = ''
        if self.package is None:
            raise ChangelogCreateError('Package not specified')
        block += self.package + ' '
        if self._raw_version is None:
            raise ChangelogCreateError('Version not specified')
        block += '(' + self._raw_version + ') '
        if self.distributions is None:
            raise ChangelogCreateError('Distribution not specified')
        block += self.distributions + '; '
        if self.urgency is None:
            raise ChangelogCreateError('Urgency not specified')
        block += 'urgency=' + self.urgency + self.urgency_comment
        for key, value in self.other_pairs.items():
            block += ', %s=%s' % (key, value)
        block += '\n'
        if self.changes() is None:
            raise ChangelogCreateError('Changes not specified')
        for change in self.changes():
            block += change + '\n'
        if not self._no_trailer:
            block += ' --'
            if self.author is not None:
                block += ' ' + self.author
            elif not allow_missing_author:
                raise ChangelogCreateError('Author not specified')
            if self.date is not None:
                block += self._trailer_separator + self.date
            elif not allow_missing_author:
                raise ChangelogCreateError('Date not specified')
            block += '\n'
        for line in self._trailing:
            block += line + '\n'
        return block

    def __str__(self):
        return self._format()

    def __bytes__(self):
        return str(self).encode(self._encoding)