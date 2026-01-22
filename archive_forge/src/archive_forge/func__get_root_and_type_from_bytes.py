import json
import typing
import warnings
from io import BytesIO
from typing import (
from warnings import warn
import jmespath
from lxml import etree, html
from packaging.version import Version
from .csstranslator import GenericTranslator, HTMLTranslator
from .utils import extract_regex, flatten, iflatten, shorten
def _get_root_and_type_from_bytes(body: bytes, encoding: str, *, input_type: Optional[str], **lxml_kwargs: Any) -> Tuple[Any, str]:
    if input_type == 'text':
        return (body.decode(encoding), input_type)
    if encoding == 'utf8':
        try:
            data = json.load(BytesIO(body))
        except ValueError:
            data = _NOT_SET
        if data is not _NOT_SET:
            return (data, 'json')
    if input_type == 'json':
        return (None, 'json')
    assert input_type in ('html', 'xml', None)
    type = _xml_or_html(input_type)
    root = create_root_node(text='', body=body, encoding=encoding, parser_cls=_ctgroup[type]['_parser'], **lxml_kwargs)
    return (root, type)