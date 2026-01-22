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