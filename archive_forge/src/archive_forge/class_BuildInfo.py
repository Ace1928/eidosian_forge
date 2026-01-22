import collections.abc
import datetime
import email.utils
import functools
import logging
import io
import re
import subprocess
import warnings
import chardet
from debian._util import (
from debian.deprecation import function_deprecated_by
import debian.debian_support
import debian.changelog
class BuildInfo(_gpg_multivalued, _PkgRelationMixin, _VersionAccessorMixin):
    """ Representation of a .buildinfo (build environment description) file

    This class is a thin wrapper around the transparent GPG handling
    of :class:`_gpg_multivalued`, the field parsing of
    :class:`_PkgRelationMixin`,
    and the format parsing of :class:`Deb822`.

    Note that the 'relations' structure returned by the `relations` method
    is identical to that produced by other classes in this module.
    Consequently, existing code to consume this structure can be used here,
    although it means that there are redundant lists and tuples within the
    structure.

    Example::

        >>> from debian.deb822 import BuildInfo
        >>> filename = 'package.buildinfo'
        >>> with open(filename) as fh:    # doctest: +SKIP
        ...     info = BuildInfo(fh)
        >>> print(info.get_environment())    # doctest: +SKIP
        {'DEB_BUILD_OPTIONS': 'parallel=4',
        'LANG': 'en_AU.UTF-8',
        'LC_ALL': 'C.UTF-8',
        'LC_TIME': 'en_GB.UTF-8',
        'LD_LIBRARY_PATH': '/usr/lib/libeatmydata',
        'SOURCE_DATE_EPOCH': '1601784586'}
        >>> installed = info.relations['installed-build-depends']  # doctest: +SKIP
        >>> for dep in installed:  # doctest: +SKIP
        ...     print("Installed %s/%s" % (dep[0]['name'], dep[0]['version'][1]))
        Installed autoconf/2.69-11.1
        Installed automake/1:1.16.2-4
        Installed autopoint/0.19.8.1-10
        Installed autotools-dev/20180224.1
        ... etc ...
        >>> changelog = info.get_changelog() # doctest: +SKIP
        >>> print(changelog.author) # doctest: +SKIP
        'xyz Build Daemon (xyz-01) <buildd_xyz-01@buildd.debian.org>'
        >>> print(changelog[0].changes()) # doctest: +SKIP
        ['',
        '  * Binary-only non-maintainer upload for amd64; no source changes.',
        '  * Add Python 3.9 as supported version',
        '']
    """
    _multivalued_fields = {'checksums-md5': ['md5', 'size', 'name'], 'checksums-sha1': ['sha1', 'size', 'name'], 'checksums-sha256': ['sha256', 'size', 'name'], 'checksums-sha512': ['sha512', 'size', 'name']}
    _relationship_fields = ['installed-build-depends']

    def __init__(self, *args, **kwargs):
        _gpg_multivalued.__init__(self, *args, **kwargs)
        _PkgRelationMixin.__init__(self, *args, **kwargs)

    def _get_array_value(self, field):
        if field not in self:
            raise KeyError("'{}' not found in buildinfo".format(field))
        return list(self[field].replace('\n', '').strip().split())

    def get_environment(self):
        """Return the build environment that was recorded

        The environment is returned as a dict in the style of `os.environ`.
        The backslash quoting of values described in deb-buildinfo(5) is
        removed.
        """
        return dict(BuildInfo._env_deserialise(self.get('Environment', '')))

    def get_changelog(self):
        """Return the changelog entry from the buildinfo (for binNMUs)

        If no "Binary-Only-Changes" field is present in the buildinfo file
        then `None` is returned.
        """
        if 'Binary-Only-Changes' not in self:
            return None
        chlines = self['Binary-Only-Changes'].splitlines()
        chlines = ['' if s == ' .' else s[1:] for s in chlines]
        return debian.changelog.Changelog(chlines)

    def get_source(self):
        if 'source' not in self:
            raise KeyError("'Source' field not found in buildinfo")
        matches = self._explicit_source_re.match(self['source'])
        if not matches:
            raise ValueError("Invalid 'Source' field specified")
        return (matches.group('source'), matches.group('version'))

    def get_binary(self):
        return self._get_array_value('Binary')

    def get_build_date(self):
        if 'build-date' not in self:
            raise KeyError("'Build-Date' field not found in buildinfo")
        timearray = email.utils.parsedate_tz(self['build-date'])
        if timearray is None:
            raise ValueError("Invalid 'Build-Date' field specified")
        ts = email.utils.mktime_tz(timearray)
        return datetime.datetime.fromtimestamp(ts)

    def get_architecture(self):
        return self._get_array_value('Architecture')

    def is_build_source(self):
        arches = [arch for arch in self.get_architecture() if arch == 'source']
        return len(arches) == 1

    def is_build_arch_all(self):
        return 'all' in self.get_architecture()

    def is_build_arch_any(self):
        arches = [arch for arch in self.get_architecture() if arch not in ('all', 'source')]
        return len(arches) == 1

    class _EnvParserState:
        IGNORE_WHITESPACE = 0
        VAR_NAME = 1
        START_VALUE_QUOTE = 2
        VALUE = 3
        VALUE_BACKSLASH_ESCAPE = 4

    @staticmethod
    def _env_deserialise(serialised):
        """ extract the environment variables and values from the text

        Format is:
            VAR_NAME="value"

        with ignorable whitespace around the construct (and separating each
        item). Quote characters within the value are backslash escaped.

        When producing the buildinfo file, dpkg only includes specifically
        allowed environment variables and thus there is no defined quoting
        rules for the variable names.

        The format is described by deb-buildinfo(5) and implemented in
        dpkg source scripts/dpkg-genbuildinfo.pl:cleansed_environment(),
        while the environment variables that are included in the output are
        listed in dpkg source scripts/Dpkg/Build/Info.pm
        """
        state = BuildInfo._EnvParserState.IGNORE_WHITESPACE
        name = ''
        value = None
        for ch in serialised:
            if state == BuildInfo._EnvParserState.IGNORE_WHITESPACE:
                if not ch.isspace():
                    state = BuildInfo._EnvParserState.VAR_NAME
                    name = ch
                continue
            if state == BuildInfo._EnvParserState.VAR_NAME:
                if ch != '=':
                    name += ch
                else:
                    state = BuildInfo._EnvParserState.START_VALUE_QUOTE
                    value = ''
                continue
            if state == BuildInfo._EnvParserState.START_VALUE_QUOTE:
                if ch == '"':
                    state = BuildInfo._EnvParserState.VALUE
                else:
                    raise ValueError('Improper quoting in Environment: begin quote not found')
                continue
            if state == BuildInfo._EnvParserState.VALUE:
                if ch == '\\':
                    state = BuildInfo._EnvParserState.VALUE_BACKSLASH_ESCAPE
                elif ch == '"':
                    if name == '':
                        raise ValueError('Improper formatting in Environment: variable name not found')
                    if value is None:
                        raise ValueError('Improper formatting in Environment: variable value not found')
                    yield (name, value)
                    state = BuildInfo._EnvParserState.IGNORE_WHITESPACE
                    name = ''
                    value = None
                else:
                    assert value is not None
                    value += ch
                continue
            if state == BuildInfo._EnvParserState.VALUE_BACKSLASH_ESCAPE:
                if ch == '"':
                    assert value is not None
                    value += ch
                    state = BuildInfo._EnvParserState.VALUE
                else:
                    raise ValueError("Improper formatting in Environment: couldn't interpret backslash sequence")
                continue
        if state != BuildInfo._EnvParserState.IGNORE_WHITESPACE:
            ValueError('Improper quoting in Environment: end quote not found')