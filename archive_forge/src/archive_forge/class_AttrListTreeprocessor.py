from __future__ import annotations
from typing import TYPE_CHECKING
from . import Extension
from ..treeprocessors import Treeprocessor
import re
class AttrListTreeprocessor(Treeprocessor):
    BASE_RE = '\\{\\:?[ ]*([^\\}\\n ][^\\n]*)[ ]*\\}'
    HEADER_RE = re.compile('[ ]+{}[ ]*$'.format(BASE_RE))
    BLOCK_RE = re.compile('\\n[ ]*{}[ ]*$'.format(BASE_RE))
    INLINE_RE = re.compile('^{}'.format(BASE_RE))
    NAME_RE = re.compile('[^A-Z_a-z\\u00c0-\\u00d6\\u00d8-\\u00f6\\u00f8-\\u02ff\\u0370-\\u037d\\u037f-\\u1fff\\u200c-\\u200d\\u2070-\\u218f\\u2c00-\\u2fef\\u3001-\\ud7ff\\uf900-\\ufdcf\\ufdf0-\\ufffd\\:\\-\\.0-9\\u00b7\\u0300-\\u036f\\u203f-\\u2040]+')

    def run(self, doc: Element) -> None:
        for elem in doc.iter():
            if self.md.is_block_level(elem.tag):
                RE = self.BLOCK_RE
                if isheader(elem) or elem.tag in ['dt', 'td', 'th']:
                    RE = self.HEADER_RE
                if len(elem) and elem.tag == 'li':
                    pos = None
                    for i, child in enumerate(elem):
                        if child.tag in ['ul', 'ol']:
                            pos = i
                            break
                    if pos is None and elem[-1].tail:
                        m = RE.search(elem[-1].tail)
                        if m:
                            if not self.assign_attrs(elem, m.group(1), strict=True):
                                elem[-1].tail = elem[-1].tail[:m.start()]
                    elif pos is not None and pos > 0 and elem[pos - 1].tail:
                        m = RE.search(elem[pos - 1].tail)
                        if m:
                            if not self.assign_attrs(elem, m.group(1), strict=True):
                                elem[pos - 1].tail = elem[pos - 1].tail[:m.start()]
                    elif elem.text:
                        m = RE.search(elem.text)
                        if m:
                            if not self.assign_attrs(elem, m.group(1), strict=True):
                                elem.text = elem.text[:m.start()]
                elif len(elem) and elem[-1].tail:
                    m = RE.search(elem[-1].tail)
                    if m:
                        if not self.assign_attrs(elem, m.group(1), strict=True):
                            elem[-1].tail = elem[-1].tail[:m.start()]
                            if isheader(elem):
                                elem[-1].tail = elem[-1].tail.rstrip('#').rstrip()
                elif elem.text:
                    m = RE.search(elem.text)
                    if m:
                        if not self.assign_attrs(elem, m.group(1), strict=True):
                            elem.text = elem.text[:m.start()]
                            if isheader(elem):
                                elem.text = elem.text.rstrip('#').rstrip()
            elif elem.tail:
                m = self.INLINE_RE.match(elem.tail)
                if m:
                    remainder = self.assign_attrs(elem, m.group(1))
                    elem.tail = elem.tail[m.end():] + remainder

    def assign_attrs(self, elem: Element, attrs_string: str, *, strict: bool=False) -> str:
        """ Assign `attrs` to element.

        If the `attrs_string` has an extra closing curly brace, the remaining text is returned.

        The `strict` argument controls whether to still assign `attrs` if there is a remaining `}`.
        """
        attrs, remainder = get_attrs_and_remainder(attrs_string)
        if strict and remainder:
            return remainder
        for k, v in attrs:
            if k == '.':
                cls = elem.get('class')
                if cls:
                    elem.set('class', '{} {}'.format(cls, v))
                else:
                    elem.set('class', v)
            else:
                elem.set(self.sanitize_name(k), v)
        return remainder

    def sanitize_name(self, name: str) -> str:
        """
        Sanitize name as 'an XML Name, minus the `:`.'
        See <https://www.w3.org/TR/REC-xml-names/#NT-NCName>.
        """
        return self.NAME_RE.sub('_', name)