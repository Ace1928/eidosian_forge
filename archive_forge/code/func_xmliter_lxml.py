import csv
import logging
import re
from io import StringIO
from typing import (
from warnings import warn
from lxml import etree
from scrapy.exceptions import ScrapyDeprecationWarning
from scrapy.http import Response, TextResponse
from scrapy.selector import Selector
from scrapy.utils.python import re_rsearch, to_unicode
def xmliter_lxml(obj: Union[Response, str, bytes], nodename: str, namespace: Optional[str]=None, prefix: str='x') -> Generator[Selector, Any, None]:
    reader = _StreamReader(obj)
    tag = f'{{{namespace}}}{nodename}' if namespace else nodename
    iterable = etree.iterparse(cast('SupportsReadClose[bytes]', reader), encoding=reader.encoding, events=('end', 'start-ns'), huge_tree=True)
    selxpath = '//' + (f'{prefix}:{nodename}' if namespace else nodename)
    needs_namespace_resolution = not namespace and ':' in nodename
    if needs_namespace_resolution:
        prefix, nodename = nodename.split(':', maxsplit=1)
    for event, data in iterable:
        if event == 'start-ns':
            assert isinstance(data, tuple)
            if needs_namespace_resolution:
                _prefix, _namespace = data
                if _prefix != prefix:
                    continue
                namespace = _namespace
                needs_namespace_resolution = False
                selxpath = f'//{prefix}:{nodename}'
                tag = f'{{{namespace}}}{nodename}'
            continue
        assert isinstance(data, etree._Element)
        node = data
        if node.tag != tag:
            continue
        nodetext = etree.tostring(node, encoding='unicode')
        node.clear()
        xs = Selector(text=nodetext, type='xml')
        if namespace:
            xs.register_namespace(prefix, namespace)
        yield xs.xpath(selxpath)[0]