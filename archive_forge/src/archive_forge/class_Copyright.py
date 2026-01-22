import collections
import itertools
import logging
import io
import re
from debian._deb822_repro import (
from debian.deb822 import RestrictedField, RestrictedFieldError
class Copyright(object):
    """Represents a debian/copyright file.

    A Copyright object contains a Header paragraph and a list of additional
    Files or License paragraphs.  It provides methods to iterate over those
    paragraphs, in addition to adding new ones.  It also provides a mechanism
    for finding the Files paragraph (if any) that matches a particular
    filename.

    Typical usage::

        with io.open('debian/copyright', 'rt', encoding='utf-8') as f:
            c = copyright.Copyright(f)

            header = c.header
            # Header exposes standard fields, e.g.
            print('Upstream name: ', header.upstream_name)
            lic = header.license
            if lic:
                print('Overall license: ', lic.synopsis)
            # You can also retrieve and set custom fields.
            header['My-Special-Field'] = 'Very special'

            # Find the license for a given file.
            paragraph = c.find_files_paragraph('debian/rules')
            if paragraph:
                print('License for debian/rules: ', paragraph.license)

            # Dump the result, including changes, to another file.
            with io.open('debian/copyright.new', 'wt', encoding='utf-8') as f:
                c.dump(f=f)

    It is possible to build up a Copyright from scratch, by modifying the
    header and using add_files_paragraph and add_license_paragraph.  See the
    associated method docstrings.
    """

    def __init__(self, sequence=None, encoding='utf-8', strict=True):
        """ Create a new copyright file in the current format.

        :param sequence: Sequence of lines, e.g. a list of strings or a
            file-like object.  If not specified, a blank Copyright object is
            initialized.
        :param encoding: Encoding to use, in case input is raw byte strings.
            It is recommended to use unicode objects everywhere instead, e.g.
            by opening files in text mode.
        :param strict: Raise if format errors are detected in the data.

        Raises:
            :class:`NotMachineReadableError` if 'sequence' does not contain a
                machine-readable debian/copyright file.
            MachineReadableFormatError if 'sequence' is not a valid file.
        """
        super(Copyright, self).__init__()
        self.__paragraphs = []
        if sequence is not None:
            header = None
            try:
                self.__file = parse_deb822_file(sequence=sequence, encoding=encoding, accept_files_with_duplicated_fields=not strict)
            except SyntaxOrParseError as e:
                raise NotMachineReadableError(str(e))
            for p in self.__file:
                if header is None:
                    header = Header(p)
                elif 'Files' in p:
                    pf = FilesParagraph(p, strict, strict=strict)
                    self.__paragraphs.append(pf)
                elif 'License' in p:
                    pl = LicenseParagraph(p, strict)
                    self.__paragraphs.append(pl)
                else:
                    _complain('Non-header paragraph has neither "Files" nor "License" fields', strict)
            if not header:
                raise NotMachineReadableError('no paragraphs in input')
            self.__header = header
        else:
            self.__file = Deb822FileElement.new_empty_file()
            self.__header = Header()
            self.__file.append(self.__header._underlying_paragraph)
            self.__paragraphs.append(self.__header)

    @property
    def header(self):
        """The file header paragraph."""
        return self.__header

    @header.setter
    def header(self, hdr):
        if not isinstance(hdr, Header):
            raise TypeError('value must be a Header object')
        self.__header = hdr

    def all_paragraphs(self):
        """Returns an iterator over all paragraphs (header, Files, License).

        The header (returned first) will be returned as a Header object; file
        paragraphs as FilesParagraph objects; license paragraphs as
        LicenseParagraph objects.

        """
        return itertools.chain([self.header], (p for p in self.__paragraphs))

    def __iter__(self):
        """Iterate over all paragraphs

        see all_paragraphs() for more information

        """
        return self.all_paragraphs()

    def all_files_paragraphs(self):
        """Returns an iterator over the contained FilesParagraph objects."""
        return (p for p in self.__paragraphs if isinstance(p, FilesParagraph))

    def find_files_paragraph(self, filename):
        """Returns the FilesParagraph for the given filename.

        In accordance with the spec, this method returns the last FilesParagraph
        that matches the filename.  If no paragraphs matched, returns None.
        """
        result = None
        for p in self.all_files_paragraphs():
            if p.matches(filename):
                result = p
        return result

    def add_files_paragraph(self, paragraph):
        """Adds a FilesParagraph to this object.

        The paragraph is inserted directly after the last FilesParagraph (which
        might be before a standalone LicenseParagraph).
        """
        if not isinstance(paragraph, FilesParagraph):
            raise TypeError('paragraph must be a FilesParagraph instance')
        last_i = -1
        for i, p in enumerate(self.__paragraphs):
            if isinstance(p, FilesParagraph):
                last_i = i
        self.__paragraphs.insert(last_i + 1, paragraph)
        self.__file.insert(last_i + 2, paragraph._underlying_paragraph)

    def all_license_paragraphs(self):
        """Returns an iterator over standalone LicenseParagraph objects."""
        return (p for p in self.__paragraphs if isinstance(p, LicenseParagraph))

    def add_license_paragraph(self, paragraph):
        """Adds a LicenceParagraph to this object.

        The paragraph is inserted after any other paragraphs.
        """
        if not isinstance(paragraph, LicenseParagraph):
            raise TypeError('paragraph must be a LicenseParagraph instance')
        self.__paragraphs.append(paragraph)
        self.__file.append(paragraph._underlying_paragraph)

    def dump(self, f=None):
        """Dumps the contents of the copyright file.

        If f is None, returns a unicode object.  Otherwise, writes the contents
        to f, which must be a file-like object that is opened in text mode
        (i.e. that accepts unicode objects directly).  It is thus up to the
        caller to arrange for the file to do any appropriate encoding.
        """
        s = self.__file.dump()
        if f is not None:
            f.write(s)
            return None
        return s