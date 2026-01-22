import datetime
import decimal
import re
from typing import (
from xml.dom.minidom import Document, parseString
from xml.dom.minidom import Element as XmlElement
from xml.parsers.expat import ExpatError
from ._utils import StreamType, deprecate_no_replacement
from .errors import PdfReadError
from .generic import ContentStream, PdfObject
def _getter_single(namespace: str, name: str, converter: Callable[[str], Any]=_identity) -> Callable[['XmpInformation'], Optional[Any]]:

    def get(self: 'XmpInformation') -> Optional[Any]:
        cached = self.cache.get(namespace, {}).get(name)
        if cached:
            return cached
        value = None
        for element in self.get_element('', namespace, name):
            if element.nodeType == element.ATTRIBUTE_NODE:
                value = element.nodeValue
            else:
                value = self._get_text(element)
            break
        if value is not None:
            value = converter(value)
        ns_cache = self.cache.setdefault(namespace, {})
        ns_cache[name] = value
        return value
    return get