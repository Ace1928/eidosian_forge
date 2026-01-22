import collections.abc
import contextlib
import sys
import textwrap
import weakref
from abc import ABC
from types import TracebackType
from weakref import ReferenceType
from debian._deb822_repro._util import (combine_into_replacement, BufferingIterator,
from debian._deb822_repro.formatter import (
from debian._deb822_repro.tokens import (
from debian._deb822_repro.types import AmbiguousDeb822FieldKeyError, SyntaxOrParseError
from debian._util import (
class Deb822FileElement(Deb822Element):
    """Represents the entire deb822 file"""

    def __init__(self, token_and_elements):
        super().__init__()
        self._token_and_elements = token_and_elements
        self._init_parent_of_parts()

    @classmethod
    def new_empty_file(cls):
        """Creates a new Deb822FileElement with no contents

        Note that a deb822 file must be non-empty to be considered valid
        """
        return cls(LinkedList())

    @property
    def is_valid_file(self):
        """Returns true if the file is valid

        Invalid elements include error elements (Deb822ErrorElement) but also
        issues such as paragraphs with duplicate fields or "empty" files
        (a valid deb822 file contains at least one paragraph).
        """
        had_paragraph = False
        for paragraph in self:
            had_paragraph = True
            if not paragraph or paragraph.has_duplicate_fields:
                return False
        if not had_paragraph:
            return False
        return self.find_first_error_element() is None

    def find_first_error_element(self):
        """Returns the first Deb822ErrorElement (or None) in the file"""
        return next(iter(self.iter_recurse(only_element_or_token_type=Deb822ErrorElement)), None)

    def __iter__(self):
        return iter(self.iter_parts_of_type(Deb822ParagraphElement))

    def iter_parts(self):
        yield from self._token_and_elements

    def insert(self, idx, para):
        """Inserts a paragraph into the file at the given "index" of paragraphs

        Note that if the index is between two paragraphs containing a "free
        floating" comment (e.g. paragrah/start-of-file, empty line, comment,
        empty line, paragraph) then it is unspecified which "side" of the
        comment the new paragraph will appear and this may change between
        versions of python-debian.


        >>> original = '''
        ... Package: libfoo-dev
        ... Depends: libfoo1 (= ${binary:Version}), ${shlib:Depends}, ${misc:Depends}
        ... '''.lstrip()
        >>> deb822_file = parse_deb822_file(original.splitlines())
        >>> para1 = Deb822ParagraphElement.new_empty_paragraph()
        >>> para1["Source"] = "foo"
        >>> para1["Build-Depends"] = "debhelper-compat (= 13)"
        >>> para2 = Deb822ParagraphElement.new_empty_paragraph()
        >>> para2["Package"] = "libfoo1"
        >>> para2["Depends"] = "${shlib:Depends}, ${misc:Depends}"
        >>> deb822_file.insert(0, para1)
        >>> deb822_file.insert(1, para2)
        >>> expected = '''
        ... Source: foo
        ... Build-Depends: debhelper-compat (= 13)
        ...
        ... Package: libfoo1
        ... Depends: ${shlib:Depends}, ${misc:Depends}
        ...
        ... Package: libfoo-dev
        ... Depends: libfoo1 (= ${binary:Version}), ${shlib:Depends}, ${misc:Depends}
        ... '''.lstrip()
        >>> deb822_file.dump() == expected
        True
        """
        anchor_node = None
        needs_newline = True
        if idx == 0:
            if not self._token_and_elements:
                self.append(para)
                return
            anchor_node = self._token_and_elements.head_node
            needs_newline = bool(self._token_and_elements)
        else:
            i = 0
            for node in self._token_and_elements.iter_nodes():
                entry = node.value
                if isinstance(entry, Deb822ParagraphElement):
                    i += 1
                if idx == i - 1:
                    anchor_node = node
                    break
        if anchor_node is None:
            self.append(para)
        else:
            if needs_newline:
                nl_token = self._set_parent(Deb822WhitespaceToken('\n'))
                anchor_node = self._token_and_elements.insert_before(nl_token, anchor_node)
            self._token_and_elements.insert_before(self._set_parent(para), anchor_node)

    def append(self, paragraph):
        """Appends a paragraph to the file

        >>> deb822_file = Deb822FileElement.new_empty_file()
        >>> para1 = Deb822ParagraphElement.new_empty_paragraph()
        >>> para1["Source"] = "foo"
        >>> para1["Build-Depends"] = "debhelper-compat (= 13)"
        >>> para2 = Deb822ParagraphElement.new_empty_paragraph()
        >>> para2["Package"] = "foo"
        >>> para2["Depends"] = "${shlib:Depends}, ${misc:Depends}"
        >>> deb822_file.append(para1)
        >>> deb822_file.append(para2)
        >>> expected = '''
        ... Source: foo
        ... Build-Depends: debhelper-compat (= 13)
        ...
        ... Package: foo
        ... Depends: ${shlib:Depends}, ${misc:Depends}
        ... '''.lstrip()
        >>> deb822_file.dump() == expected
        True
        """
        tail_element = self._token_and_elements.tail
        if paragraph.parent_element is not None:
            if paragraph.parent_element is self:
                raise ValueError('Paragraph is already a part of this file')
            raise ValueError('Paragraph is already part of another Deb822File')
        if tail_element and (not isinstance(tail_element, Deb822WhitespaceToken)):
            self._token_and_elements.append(self._set_parent(Deb822WhitespaceToken('\n')))
        self._token_and_elements.append(self._set_parent(paragraph))

    def remove(self, paragraph):
        if paragraph.parent_element is not self:
            raise ValueError('Paragraph is part of a different file')
        node = None
        for node in self._token_and_elements.iter_nodes():
            if node.value is paragraph:
                break
        if node is None:
            raise RuntimeError('unable to find paragraph')
        previous_node = node.previous_node
        next_node = node.next_node
        self._token_and_elements.remove_node(node)
        if next_node is None:
            if previous_node and isinstance(previous_node.value, Deb822WhitespaceToken):
                self._token_and_elements.remove_node(previous_node)
        elif isinstance(next_node.value, Deb822WhitespaceToken):
            self._token_and_elements.remove_node(next_node)
        paragraph.parent_element = None

    def _set_parent(self, t):
        t.parent_element = self
        return t

    @overload
    def dump(self, fd):
        pass

    @overload
    def dump(self):
        pass

    def dump(self, fd=None):
        if fd is None:
            return ''.join((t.text for t in self.iter_tokens()))
        for token in self.iter_tokens():
            fd.write(token.text.encode('utf-8'))
        return None