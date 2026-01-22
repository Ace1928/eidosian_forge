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
def _css2xpath(self, query: str) -> str:
    type = _xml_or_html(self.type)
    return _ctgroup[type]['_csstranslator'].css_to_xpath(query)