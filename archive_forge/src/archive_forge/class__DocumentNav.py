from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
class _DocumentNav:
    """Navigate a Beautiful Soup document."""

    @classmethod
    def assert_valid_input(cls, tag: Any) -> None:
        """Check if valid input tag or document."""
        if not cls.is_tag(tag):
            raise TypeError(f"Expected a BeautifulSoup 'Tag', but instead received type {type(tag)}")

    @staticmethod
    def is_doc(obj: bs4.Tag) -> bool:
        """Is `BeautifulSoup` object."""
        return isinstance(obj, bs4.BeautifulSoup)

    @staticmethod
    def is_tag(obj: bs4.PageElement) -> bool:
        """Is tag."""
        return isinstance(obj, bs4.Tag)

    @staticmethod
    def is_declaration(obj: bs4.PageElement) -> bool:
        """Is declaration."""
        return isinstance(obj, bs4.Declaration)

    @staticmethod
    def is_cdata(obj: bs4.PageElement) -> bool:
        """Is CDATA."""
        return isinstance(obj, bs4.CData)

    @staticmethod
    def is_processing_instruction(obj: bs4.PageElement) -> bool:
        """Is processing instruction."""
        return isinstance(obj, bs4.ProcessingInstruction)

    @staticmethod
    def is_navigable_string(obj: bs4.PageElement) -> bool:
        """Is navigable string."""
        return isinstance(obj, bs4.NavigableString)

    @staticmethod
    def is_special_string(obj: bs4.PageElement) -> bool:
        """Is special string."""
        return isinstance(obj, (bs4.Comment, bs4.Declaration, bs4.CData, bs4.ProcessingInstruction, bs4.Doctype))

    @classmethod
    def is_content_string(cls, obj: bs4.PageElement) -> bool:
        """Check if node is content string."""
        return cls.is_navigable_string(obj) and (not cls.is_special_string(obj))

    @staticmethod
    def create_fake_parent(el: bs4.Tag) -> _FakeParent:
        """Create fake parent for a given element."""
        return _FakeParent(el)

    @staticmethod
    def is_xml_tree(el: bs4.Tag) -> bool:
        """Check if element (or document) is from a XML tree."""
        return bool(el._is_xml)

    def is_iframe(self, el: bs4.Tag) -> bool:
        """Check if element is an `iframe`."""
        return bool((el.name if self.is_xml_tree(el) else util.lower(el.name)) == 'iframe' and self.is_html_tag(el))

    def is_root(self, el: bs4.Tag) -> bool:
        """
        Return whether element is a root element.

        We check that the element is the root of the tree (which we have already pre-calculated),
        and we check if it is the root element under an `iframe`.
        """
        root = self.root and self.root is el
        if not root:
            parent = self.get_parent(el)
            root = parent is not None and self.is_html and self.is_iframe(parent)
        return root

    def get_contents(self, el: bs4.Tag, no_iframe: bool=False) -> Iterator[bs4.PageElement]:
        """Get contents or contents in reverse."""
        if not no_iframe or not self.is_iframe(el):
            yield from el.contents

    def get_children(self, el: bs4.Tag, start: int | None=None, reverse: bool=False, tags: bool=True, no_iframe: bool=False) -> Iterator[bs4.PageElement]:
        """Get children."""
        if not no_iframe or not self.is_iframe(el):
            last = len(el.contents) - 1
            if start is None:
                index = last if reverse else 0
            else:
                index = start
            end = -1 if reverse else last + 1
            incr = -1 if reverse else 1
            if 0 <= index <= last:
                while index != end:
                    node = el.contents[index]
                    index += incr
                    if not tags or self.is_tag(node):
                        yield node

    def get_descendants(self, el: bs4.Tag, tags: bool=True, no_iframe: bool=False) -> Iterator[bs4.PageElement]:
        """Get descendants."""
        if not no_iframe or not self.is_iframe(el):
            next_good = None
            for child in el.descendants:
                if next_good is not None:
                    if child is not next_good:
                        continue
                    next_good = None
                is_tag = self.is_tag(child)
                if no_iframe and is_tag and self.is_iframe(child):
                    if child.next_sibling is not None:
                        next_good = child.next_sibling
                    else:
                        last_child = child
                        while self.is_tag(last_child) and last_child.contents:
                            last_child = last_child.contents[-1]
                        next_good = last_child.next_element
                    yield child
                    if next_good is None:
                        break
                    continue
                if not tags or is_tag:
                    yield child

    def get_parent(self, el: bs4.Tag, no_iframe: bool=False) -> bs4.Tag:
        """Get parent."""
        parent = el.parent
        if no_iframe and parent is not None and self.is_iframe(parent):
            parent = None
        return parent

    @staticmethod
    def get_tag_name(el: bs4.Tag) -> str | None:
        """Get tag."""
        return cast('str | None', el.name)

    @staticmethod
    def get_prefix_name(el: bs4.Tag) -> str | None:
        """Get prefix."""
        return cast('str | None', el.prefix)

    @staticmethod
    def get_uri(el: bs4.Tag) -> str | None:
        """Get namespace `URI`."""
        return cast('str | None', el.namespace)

    @classmethod
    def get_next(cls, el: bs4.Tag, tags: bool=True) -> bs4.PageElement:
        """Get next sibling tag."""
        sibling = el.next_sibling
        while tags and (not cls.is_tag(sibling)) and (sibling is not None):
            sibling = sibling.next_sibling
        return sibling

    @classmethod
    def get_previous(cls, el: bs4.Tag, tags: bool=True) -> bs4.PageElement:
        """Get previous sibling tag."""
        sibling = el.previous_sibling
        while tags and (not cls.is_tag(sibling)) and (sibling is not None):
            sibling = sibling.previous_sibling
        return sibling

    @staticmethod
    def has_html_ns(el: bs4.Tag) -> bool:
        """
        Check if element has an HTML namespace.

        This is a bit different than whether a element is treated as having an HTML namespace,
        like we do in the case of `is_html_tag`.
        """
        ns = getattr(el, 'namespace') if el else None
        return bool(ns and ns == NS_XHTML)

    @staticmethod
    def split_namespace(el: bs4.Tag, attr_name: str) -> tuple[str | None, str | None]:
        """Return namespace and attribute name without the prefix."""
        return (getattr(attr_name, 'namespace', None), getattr(attr_name, 'name', None))

    @classmethod
    def normalize_value(cls, value: Any) -> str | Sequence[str]:
        """Normalize the value to be a string or list of strings."""
        if value is None:
            return ''
        if isinstance(value, str):
            return value
        if isinstance(value, bytes):
            return value.decode('utf8')
        if isinstance(value, Sequence):
            new_value = []
            for v in value:
                if not isinstance(v, (str, bytes)) and isinstance(v, Sequence):
                    new_value.append(str(v))
                else:
                    new_value.append(cast(str, cls.normalize_value(v)))
            return new_value
        return str(value)

    @classmethod
    def get_attribute_by_name(cls, el: bs4.Tag, name: str, default: str | Sequence[str] | None=None) -> str | Sequence[str] | None:
        """Get attribute by name."""
        value = default
        if el._is_xml:
            try:
                value = cls.normalize_value(el.attrs[name])
            except KeyError:
                pass
        else:
            for k, v in el.attrs.items():
                if util.lower(k) == name:
                    value = cls.normalize_value(v)
                    break
        return value

    @classmethod
    def iter_attributes(cls, el: bs4.Tag) -> Iterator[tuple[str, str | Sequence[str] | None]]:
        """Iterate attributes."""
        for k, v in el.attrs.items():
            yield (k, cls.normalize_value(v))

    @classmethod
    def get_classes(cls, el: bs4.Tag) -> Sequence[str]:
        """Get classes."""
        classes = cls.get_attribute_by_name(el, 'class', [])
        if isinstance(classes, str):
            classes = RE_NOT_WS.findall(classes)
        return cast(Sequence[str], classes)

    def get_text(self, el: bs4.Tag, no_iframe: bool=False) -> str:
        """Get text."""
        return ''.join([node for node in self.get_descendants(el, tags=False, no_iframe=no_iframe) if self.is_content_string(node)])

    def get_own_text(self, el: bs4.Tag, no_iframe: bool=False) -> list[str]:
        """Get Own Text."""
        return [node for node in self.get_contents(el, no_iframe=no_iframe) if self.is_content_string(node)]