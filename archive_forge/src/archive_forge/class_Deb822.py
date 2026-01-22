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
class Deb822(Deb822Dict):
    """ Generic Deb822 data

    :param sequence: a string, or any object that returns a line of
        input each time, normally a file.  Alternately, sequence can
        be a dict that contains the initial key-value pairs. When
        python-apt is present, sequence can also be a compressed object,
        for example a file object associated to something.gz.

    :param fields: if given, it is interpreted as a list of fields that
        should be parsed (the rest will be discarded).

    :param _parsed: internal parameter.

    :param encoding: When parsing strings, interpret them in this encoding.
        (All values are given back as unicode objects, so an encoding is
        necessary in order to properly interpret the strings.)

    :param strict: Dict controlling the strictness of the internal parser
        to permit tuning of its behaviour between "generous in what it
        accepts" and "strict conformance". Known keys are described below.

    *Internal parser tuning*

    - `whitespace-separates-paragraphs`: (default: `True`)
      Blank lines between paragraphs should not have any whitespace in them
      at all. However:

      - Policy §5.1 permits `debian/control` in source packages to separate
        packages with lines containing whitespace to allow human edited
        files to have stray whitespace. Failing to honour this breaks
        tools such as
        `wrap-and-sort <https://manpages.debian.org/wrap-and-sort>`_
        (see, for example,
        `Debian Bug 715558 <https://bugs.debian.org/715558/>`_).
      - `apt_pkg.TagFile` accepts whitespace-only lines within the
        `Description` field; strictly matching the behaviour of apt's
        Deb822 parser requires setting this key to `False` (as is done
        by default for :class:`Sources` and :class:`Packages`.
        (see, for example,
        `Debian Bug 913274 <https://bugs.debian.org/913274/>`_).

    Note that these tuning parameter are only for the parser that is
    internal to `Deb822` and do not apply to python-apt's apt_pkg.TagFile
    parser which would normally be used for Packages and Sources files.
    """

    def __init__(self, sequence=None, fields=None, _parsed=None, encoding='utf-8', strict=None):
        _dict = {}
        iterable = None
        if hasattr(sequence, 'items'):
            _dict = cast(Deb822Mapping, sequence)
        else:
            iterable = cast(InputDataType, sequence)
        Deb822Dict.__init__(self, _dict=_dict, _parsed=_parsed, _fields=fields, encoding=encoding)
        if iterable is not None:
            try:
                self._internal_parser(iterable, fields, strict)
            except EOFError:
                pass
        self.gpg_info = None
    if TYPE_CHECKING:
        T_Deb822 = TypeVar('T_Deb822', bound='Deb822')

    @classmethod
    def iter_paragraphs(cls, sequence, fields=None, use_apt_pkg=False, shared_storage=False, encoding='utf-8', strict=None):
        """Generator that yields a Deb822 object for each paragraph in sequence.

        :param sequence: same as in __init__.

        :param fields: likewise.

        :param use_apt_pkg: if sequence is a file, apt_pkg can be used
            if available to parse the file, since it's much much faster.  Set
            this parameter to True to enable use of apt_pkg. Note that the
            TagFile parser from apt_pkg is a much stricter parser of the
            Deb822 format, particularly with regards whitespace between
            paragraphs and comments within paragraphs. If these features are
            required (for example in debian/control files), ensure that this
            parameter is set to False.
        :param shared_storage: not used, here for historical reasons.  Deb822
            objects never use shared storage anymore.
        :param encoding: Interpret the paragraphs in this encoding.
            (All values are given back as unicode objects, so an encoding is
            necessary in order to properly interpret the strings.)
        :param strict: dict of settings to tune the internal parser if that is
            being used. See the documentation for :class:`Deb822` for details.
        """
        apt_pkg_allowed = use_apt_pkg and _has_fileno(sequence)
        if use_apt_pkg and (not _have_apt_pkg):
            msg = "Parsing of Deb822 data with python3-apt's apt_pkg was requested but this package is not importable. Is python3-apt installed?"
            warnings.warn(msg)
        elif use_apt_pkg and (not apt_pkg_allowed):
            msg = "Parsing of Deb822 data with python3-apt's apt_pkg was requested but this cannot be done on non-file input."
            warnings.warn(msg)
        if _have_apt_pkg and apt_pkg_allowed:
            parser = apt_pkg.TagFile(sequence, bytes=True)
            for section in parser:
                paragraph = cls(fields=fields, _parsed=TagSectionWrapper(section, _AutoDecoder(encoding)), encoding=encoding)
                if paragraph:
                    yield paragraph
        else:
            iterable = []
            if isinstance(sequence, str):
                iterable = iter(sequence.splitlines())
            elif isinstance(sequence, bytes):
                iterable = iter(sequence.splitlines())
            else:
                iterable = iter(sequence)
            while True:
                x = cls(iterable, fields, encoding=encoding, strict=strict)
                if not x:
                    break
                yield x

    @staticmethod
    def _skip_useless_lines(sequence):
        """Yields only lines that do not begin with '#'.

        Also skips any blank lines at the beginning of the input.
        """
        at_beginning = True
        for line in sequence:
            if isinstance(line, str):
                line = line.encode()
            if line.startswith(b'#'):
                continue
            if at_beginning:
                if not line.rstrip(b'\r\n'):
                    continue
                at_beginning = False
            yield line
    _key_part = '^(?P<key>[^: \\t\\n\\r\\f\\v]+)\\s*:\\s*'
    _new_field_re = re.compile(_key_part + '(?P<data>(?:\\S+(\\s+\\S+)*)?)\\s*$')
    _explicit_source_re = re.compile('(?P<source>[^ ]+)( \\((?P<version>.+)\\))?')

    def _internal_parser(self, sequence, fields=None, strict=None):

        def wanted_field(f):
            return fields is None or f in fields
        if isinstance(sequence, (str, bytes)):
            sequence = sequence.splitlines()
        curkey = None
        content = ''
        for linebytes in self._gpg_stripped_paragraph(self._skip_useless_lines(sequence), strict):
            line = self.decoder.decode_bytes(linebytes)
            m = self._new_field_re.match(line)
            if m:
                if curkey:
                    self[curkey] = content
                curkey = m.group('key')
                if not wanted_field(curkey):
                    curkey = None
                    continue
                content = m.group('data')
                continue
            if line and line[0].isspace() and (not line.isspace()):
                content += '\n' + line
                continue
        if curkey:
            self[curkey] = content

    def __str__(self):
        d = self.dump()
        return d if d is not None else ''

    def __unicode__(self):
        d = self.dump()
        return d if d is not None else ''

    def __bytes__(self):
        d = self.dump()
        return d.encode(self.encoding) if d is not None else b''

    def get_as_string(self, key):
        """Return the self[key] as a string (or unicode)

        The default implementation just returns unicode(self[key]); however,
        this can be overridden in subclasses (e.g. _multivalued) that can take
        special values.
        """
        return str(self[key])

    def _dump_format(self):
        for key in self:
            value = self.get_as_string(key)
            if not value or value[0] == '\n':
                entry = '%s:%s\n' % (key, value)
            else:
                entry = '%s: %s\n' % (key, value)
            yield entry

    def _dump_str(self):
        return ''.join(self._dump_format())

    def _dump_fd_b(self, fd, encoding):
        for entry in self._dump_format():
            fd.write(entry.encode(encoding))

    def _dump_fd_t(self, fd):
        for entry in self._dump_format():
            fd.write(entry)

    @overload
    def dump(self):
        pass

    @overload
    def dump(self, fd, encoding=None, text_mode=False):
        pass

    @overload
    def dump(self, fd, encoding=None, text_mode=True):
        pass

    @overload
    def dump(self, fd=None, encoding=None, text_mode=False):
        pass

    @overload
    def dump(self, fd=None, encoding=None, text_mode=False):
        pass

    def dump(self, fd=None, encoding=None, text_mode=False):
        """Dump the contents in the original format

        :param fd: file-like object to which the data should be written
            (see notes below)
        :param encoding: str, optional (Defaults to object default).
            Encoding to use when writing out the data.
        :param text_mode: bool, optional (Defaults to ``False``).
            Encoding should be undertaken by this function rather than by the
            caller.

        If fd is None, returns a unicode object.  Otherwise, fd is assumed to
        be a file-like object, and this method will write the data to it
        instead of returning a unicode object.

        If fd is not none and text_mode is False, the data will be encoded
        to a byte string before writing to the file.  The encoding used is
        chosen via the encoding parameter; None means to use the encoding the
        object was initialized with (utf-8 by default).  This will raise
        UnicodeEncodeError if the encoding can't support all the characters in
        the Deb822Dict values.
        """
        if fd is None:
            return self._dump_str()
        if text_mode:
            self._dump_fd_t(cast(IO[str], fd))
        else:
            if encoding is None:
                encoding = self.encoding
            self._dump_fd_b(cast(IO[bytes], fd), encoding)
        return None

    @staticmethod
    def is_single_line(s):
        return not s.count('\n')
    isSingleLine = function_deprecated_by(is_single_line)

    @staticmethod
    def is_multi_line(s):
        return not Deb822.is_single_line(s)
    isMultiLine = function_deprecated_by(is_multi_line)

    def _merge_fields(self, s1, s2):
        if not s2:
            return s1
        if not s1:
            return s2
        if self.is_single_line(s1) and self.is_single_line(s2):
            delim = ' '
            if (s1 + s2).count(', '):
                delim = ', '
            L = sorted((s1 + delim + s2).split(delim))
            prev = merged = L[0]
            for item in L[1:]:
                if item == prev:
                    continue
                merged = merged + delim + item
                prev = item
            return merged
        if self.is_multi_line(s1) and self.is_multi_line(s2):
            for item in s2.splitlines(True):
                if item not in s1.splitlines(True):
                    s1 = s1 + '\n' + item
            return s1
        raise ValueError
    _mergeFields = function_deprecated_by(_merge_fields)

    def merge_fields(self, key, d1, d2=None):
        if d2 is None:
            x1 = self
            x2 = d1
        else:
            x1 = d1
            x2 = d2
        if key in x1 and key in x2:
            merged = self._merge_fields(x1[key], x2[key])
        elif key in x1:
            merged = x1[key]
        elif key in x2:
            merged = x2[key]
        else:
            raise KeyError
        if d2 is None:
            self[key] = merged
            return None
        return merged
    mergeFields = function_deprecated_by(merge_fields)
    _gpgre = re.compile(b'^-----(?P<action>BEGIN|END) PGP (?P<what>[^-]+)-----[\\r\\t ]*$')

    @staticmethod
    def split_gpg_and_payload(sequence, strict=None):
        """Return a (gpg_pre, payload, gpg_post) tuple

        Each element of the returned tuple is a list of lines (with trailing
        whitespace stripped).

        :param sequence: iterable.
            An iterable that yields lines of data (str, unicode,
            bytes) to be parsed, possibly including a GPG in-line signature.
        :param strict: dict, optional.
            Control over the strictness of the parser. See the :class:`Deb822`
            class documentation for details.
        """
        _encoded_sequence = (x.encode() if isinstance(x, str) else x for x in sequence)
        return Deb822._split_gpg_and_payload(_encoded_sequence, strict=strict)

    @staticmethod
    def _split_gpg_and_payload(sequence, strict=None):
        if not strict:
            strict = {}
        gpg_pre_lines = []
        lines = []
        gpg_post_lines = []
        state = b'SAFE'
        accept_empty_or_whitespace = strict.get('whitespace-separates-paragraphs', True)
        first_line = True
        for line in sequence:
            line = line.strip(b'\r\n')
            if first_line:
                if not line or line.isspace():
                    continue
                first_line = False
            m = Deb822._gpgre.match(line) if line.startswith(b'-') else None
            is_empty_line = not line or line.isspace() if accept_empty_or_whitespace else not line
            if not m:
                if state == b'SAFE':
                    if not is_empty_line:
                        lines.append(line)
                    elif not gpg_pre_lines:
                        break
                elif state == b'SIGNED MESSAGE':
                    if is_empty_line:
                        state = b'SAFE'
                    else:
                        gpg_pre_lines.append(line)
                elif state == b'SIGNATURE':
                    gpg_post_lines.append(line)
            else:
                if m.group('action') == b'BEGIN':
                    state = m.group('what')
                elif m.group('action') == b'END':
                    gpg_post_lines.append(line)
                    break
                if not is_empty_line:
                    if not lines:
                        gpg_pre_lines.append(line)
                    else:
                        gpg_post_lines.append(line)
        if lines:
            return (gpg_pre_lines, lines, gpg_post_lines)
        raise EOFError('only blank lines found in input')

    @classmethod
    def _gpg_stripped_paragraph(cls, sequence, strict=None):
        return cls._split_gpg_and_payload(sequence, strict)[1]

    def get_gpg_info(self, keyrings=None):
        """Return a GpgInfo object with GPG signature information

        This method will raise ValueError if the signature is not available
        (e.g. the original text cannot be found).

        :param keyrings: list of keyrings to use (see GpgInfo.from_sequence)
        """
        if not hasattr(self, 'raw_text'):
            raise ValueError('original text cannot be found')
        if self.gpg_info is None:
            self.gpg_info = GpgInfo.from_sequence(self.raw_text, keyrings=keyrings)
        return self.gpg_info

    def validate_input(self, key, value):
        """Raise ValueError if value is not a valid value for key

        Subclasses that do interesting things for different keys may wish to
        override this method.
        """
        if '\n' not in value:
            return
        if value.endswith('\n'):
            raise ValueError("value must not end in '\\n'")
        for no, line in enumerate(value.splitlines()):
            if no == 0:
                continue
            if not line:
                raise ValueError('value must not have blank lines')
            if not line[0].isspace():
                raise ValueError('each line must start with whitespace')

    def __setitem__(self, key, value):
        self.validate_input(key, value)
        Deb822Dict.__setitem__(self, key, value)