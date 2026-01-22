from __future__ import annotations
from collections import abc
import numbers
import re
from re import Pattern
from typing import (
import warnings
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend
from pandas.core.dtypes.common import is_list_like
from pandas import isna
from pandas.core.indexes.base import Index
from pandas.core.indexes.multi import MultiIndex
from pandas.core.series import Series
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import (
from pandas.io.formats.printing import pprint_thing
from pandas.io.parsers import TextParser
class _LxmlFrameParser(_HtmlFrameParser):
    """
    HTML to DataFrame parser that uses lxml under the hood.

    Warning
    -------
    This parser can only handle HTTP, FTP, and FILE urls.

    See Also
    --------
    _HtmlFrameParser
    _BeautifulSoupLxmlFrameParser

    Notes
    -----
    Documentation strings for this class are in the base class
    :class:`_HtmlFrameParser`.
    """

    def _href_getter(self, obj) -> str | None:
        href = obj.xpath('.//a/@href')
        return None if not href else href[0]

    def _text_getter(self, obj):
        return obj.text_content()

    def _parse_td(self, row):
        return row.xpath('./td|./th')

    def _parse_tables(self, document, match, kwargs):
        pattern = match.pattern
        xpath_expr = f'//table[.//text()[re:test(., {repr(pattern)})]]'
        if kwargs:
            xpath_expr += _build_xpath_expr(kwargs)
        tables = document.xpath(xpath_expr, namespaces=_re_namespace)
        tables = self._handle_hidden_tables(tables, 'attrib')
        if self.displayed_only:
            for table in tables:
                for elem in table.xpath('.//style'):
                    elem.drop_tree()
                for elem in table.xpath('.//*[@style]'):
                    if 'display:none' in elem.attrib.get('style', '').replace(' ', ''):
                        elem.drop_tree()
        if not tables:
            raise ValueError(f'No tables found matching regex {repr(pattern)}')
        return tables

    def _equals_tag(self, obj, tag) -> bool:
        return obj.tag == tag

    def _build_doc(self):
        """
        Raises
        ------
        ValueError
            * If a URL that lxml cannot parse is passed.

        Exception
            * Any other ``Exception`` thrown. For example, trying to parse a
              URL that is syntactically correct on a machine with no internet
              connection will fail.

        See Also
        --------
        pandas.io.html._HtmlFrameParser._build_doc
        """
        from lxml.etree import XMLSyntaxError
        from lxml.html import HTMLParser, fromstring, parse
        parser = HTMLParser(recover=True, encoding=self.encoding)
        try:
            if is_url(self.io):
                with get_handle(self.io, 'r', storage_options=self.storage_options) as f:
                    r = parse(f.handle, parser=parser)
            else:
                r = parse(self.io, parser=parser)
            try:
                r = r.getroot()
            except AttributeError:
                pass
        except (UnicodeDecodeError, OSError) as e:
            if not is_url(self.io):
                r = fromstring(self.io, parser=parser)
                try:
                    r = r.getroot()
                except AttributeError:
                    pass
            else:
                raise e
        else:
            if not hasattr(r, 'text_content'):
                raise XMLSyntaxError('no text parsed from document', 0, 0, 0)
        for br in r.xpath('*//br'):
            br.tail = '\n' + (br.tail or '')
        return r

    def _parse_thead_tr(self, table):
        rows = []
        for thead in table.xpath('.//thead'):
            rows.extend(thead.xpath('./tr'))
            elements_at_root = thead.xpath('./td|./th')
            if elements_at_root:
                rows.append(thead)
        return rows

    def _parse_tbody_tr(self, table):
        from_tbody = table.xpath('.//tbody//tr')
        from_root = table.xpath('./tr')
        return from_tbody + from_root

    def _parse_tfoot_tr(self, table):
        return table.xpath('.//tfoot//tr')