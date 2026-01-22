from __future__ import annotations
import codecs
import io
from typing import (
import warnings
from pandas.errors import AbstractMethodError
from pandas.util._decorators import (
from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.missing import isna
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import get_handle
from pandas.io.xml import (
class LxmlXMLFormatter(_BaseXMLFormatter):
    """
    Class for formatting data in xml using Python standard library
    modules: `xml.etree.ElementTree` and `xml.dom.minidom`.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._convert_empty_str_key()

    def _build_tree(self) -> bytes:
        """
        Build tree from  data.

        This method initializes the root and builds attributes and elements
        with optional namespaces.
        """
        from lxml.etree import Element, SubElement, tostring
        self.root = Element(f'{self.prefix_uri}{self.root_name}', nsmap=self.namespaces)
        for d in self.frame_dicts.values():
            elem_row = SubElement(self.root, f'{self.prefix_uri}{self.row_name}')
            if not self.attr_cols and (not self.elem_cols):
                self.elem_cols = list(d.keys())
                self._build_elems(d, elem_row)
            else:
                elem_row = self._build_attribs(d, elem_row)
                self._build_elems(d, elem_row)
        self.out_xml = tostring(self.root, pretty_print=self.pretty_print, method='xml', encoding=self.encoding, xml_declaration=self.xml_declaration)
        if self.stylesheet is not None:
            self.out_xml = self._transform_doc()
        return self.out_xml

    def _convert_empty_str_key(self) -> None:
        """
        Replace zero-length string in `namespaces`.

        This method will replace '' with None to align to `lxml`
        requirement that empty string prefixes are not allowed.
        """
        if self.namespaces and '' in self.namespaces.keys():
            self.namespaces[None] = self.namespaces.pop('', 'default')

    def _get_prefix_uri(self) -> str:
        uri = ''
        if self.namespaces:
            if self.prefix:
                try:
                    uri = f'{{{self.namespaces[self.prefix]}}}'
                except KeyError:
                    raise KeyError(f'{self.prefix} is not included in namespaces')
            elif '' in self.namespaces:
                uri = f'{{{self.namespaces['']}}}'
            else:
                uri = ''
        return uri

    @cache_readonly
    def _sub_element_cls(self):
        from lxml.etree import SubElement
        return SubElement

    def _transform_doc(self) -> bytes:
        """
        Parse stylesheet from file or buffer and run it.

        This method will parse stylesheet object into tree for parsing
        conditionally by its specific object type, then transforms
        original tree with XSLT script.
        """
        from lxml.etree import XSLT, XMLParser, fromstring, parse
        style_doc = self.stylesheet
        assert style_doc is not None
        handle_data = get_data_from_filepath(filepath_or_buffer=style_doc, encoding=self.encoding, compression=self.compression, storage_options=self.storage_options)
        with preprocess_data(handle_data) as xml_data:
            curr_parser = XMLParser(encoding=self.encoding)
            if isinstance(xml_data, io.StringIO):
                xsl_doc = fromstring(xml_data.getvalue().encode(self.encoding), parser=curr_parser)
            else:
                xsl_doc = parse(xml_data, parser=curr_parser)
        transformer = XSLT(xsl_doc)
        new_doc = transformer(self.root)
        return bytes(new_doc)