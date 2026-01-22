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
def _get_root_type(root: Any, *, input_type: Optional[str]) -> str:
    if isinstance(root, etree._Element):
        if input_type in {'json', 'text'}:
            raise ValueError(f'Selector got an lxml.etree._Element object as root, and {input_type!r} as type.')
        return _xml_or_html(input_type)
    elif isinstance(root, (dict, list)) or _is_valid_json(root):
        return 'json'
    return input_type or 'json'