import json
from decimal import Decimal, ROUND_UP
from types import ModuleType
from typing import cast, Any, Dict, Iterator, Iterable, Optional, Set, Union, Tuple
from xml.etree import ElementTree
from .exceptions import ElementPathError, xpath_error
from .namespaces import XSLT_XQUERY_SERIALIZATION_NAMESPACE
from .datatypes import AnyAtomicType, AnyURI, AbstractDateTime, \
from .xpath_nodes import XPathNode, ElementNode, AttributeNode, DocumentNode, \
from .xpath_tokens import XPathToken, XPathMap, XPathArray
from .protocols import EtreeElementProtocol, LxmlElementProtocol
class XPathEncoder(json.JSONEncoder):

    def default(self, obj: Any) -> Any:
        if isinstance(obj, XPathNode):
            if isinstance(obj, DocumentNode):
                return ''.join((self.default(child) for child in obj))
            elif isinstance(obj, ElementNode):
                elem = obj.elem
                assert etree_module is not None
                try:
                    chunks = etree_module.tostringlist(elem, encoding='utf-8')
                except TypeError:
                    chunk = etree_module.tostring(elem, encoding='utf-8')
                    return cast(str, chunk.decode('utf-8'))
                else:
                    if chunks and chunks[0].startswith(b'<?'):
                        chunks[0] = chunks[0].replace(b"'", b'"')
                    return b'\n'.join(chunks).decode('utf-8')
            elif isinstance(obj, (AttributeNode, NamespaceNode)):
                return f'{obj.name}="{obj.string_value}"'
            elif isinstance(obj, TextNode):
                return obj.value
            elif isinstance(obj, CommentNode):
                return f'<!--{obj.string_value}-->'
            else:
                return f'<?{obj.name} {obj.string_value}?>'
        elif isinstance(obj, XPathMap):
            if any((isinstance(v, list) and len(v) > 1 for v in obj.values())):
                raise xpath_error('SERE0023', token=token)
            map_keys = set()
            map_items = []
            k: Any
            for k, v in obj.items():
                if isinstance(k, QName):
                    k = str(k)
                map_items.append((k, v))
                if k not in map_keys:
                    map_keys.add(k)
                elif not params.get('allow_duplicate_names'):
                    raise xpath_error('SERE0022', token=token)
            return MapEncodingDict(map_items)
        elif isinstance(obj, XPathArray):
            return [v if v or not isinstance(v, list) else None for v in obj.items()]
        elif isinstance(obj, (AbstractBinary, AbstractDateTime, AnyURI, UntypedAtomic)):
            return str(obj)
        elif isinstance(obj, Decimal):
            return float(Decimal(obj).quantize(Decimal('0.01'), ROUND_UP))
        else:
            return super().default(obj)