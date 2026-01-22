import sys
from abc import abstractmethod
from typing import cast, overload, Any, Dict, Iterator, List, Optional, \
import re
from elementpath import XPath2Parser, XPathSchemaContext, \
from .exceptions import XMLSchemaValueError, XMLSchemaTypeError
from .names import XSD_NAMESPACE
from .aliases import NamespacesType, SchemaType, BaseXsdType, XPathElementType
from .helpers import get_qname, local_name, get_prefixed_qname
class XMLSchemaProxy(AbstractSchemaProxy):
    """XPath schema proxy for the *xmlschema* library."""
    _schema: SchemaType

    def __init__(self, schema: Optional[XsdSchemaProtocol]=None, base_element: Optional[XsdElementProtocol]=None) -> None:
        if schema is None:
            from xmlschema import XMLSchema10
            schema = cast(XsdSchemaProtocol, getattr(XMLSchema10, 'meta_schema', None))
        super(XMLSchemaProxy, self).__init__(schema, base_element)
        if base_element is not None:
            try:
                if base_element.schema is not schema:
                    msg = '{} is not an element of {}'
                    raise XMLSchemaValueError(msg.format(base_element, schema))
            except AttributeError:
                raise XMLSchemaTypeError('%r is not an XsdElement' % base_element)

    def bind_parser(self, parser: XPath2Parser) -> None:
        parser.schema = self
        parser.symbol_table = dict(parser.__class__.symbol_table)
        with self._schema.lock:
            if self._schema.xpath_tokens is None:
                self._schema.xpath_tokens = {xsd_type.name: parser.schema_constructor(xsd_type.name) for xsd_type in self.iter_atomic_types() if xsd_type.name}
        parser.symbol_table.update(self._schema.xpath_tokens)

    def get_context(self) -> XPathSchemaContext:
        return XPathSchemaContext(root=self._schema.xpath_node, namespaces=dict(self._schema.namespaces), item=self._base_element)

    def is_instance(self, obj: Any, type_qname: str) -> bool:
        xsd_type = self._schema.maps.types[type_qname]
        if isinstance(xsd_type, tuple):
            from .validators import XMLSchemaNotBuiltError
            schema = xsd_type[1]
            raise XMLSchemaNotBuiltError(schema, f'XSD type {type_qname!r} is not built')
        try:
            xsd_type.encode(obj)
        except ValueError:
            return False
        else:
            return True

    def cast_as(self, obj: Any, type_qname: str) -> Any:
        xsd_type = self._schema.maps.types[type_qname]
        if isinstance(xsd_type, tuple):
            from .validators import XMLSchemaNotBuiltError
            schema = xsd_type[1]
            raise XMLSchemaNotBuiltError(schema, f'XSD type {type_qname!r} is not built')
        return xsd_type.decode(obj)

    def iter_atomic_types(self) -> Iterator[XsdTypeProtocol]:
        for xsd_type in self._schema.maps.types.values():
            if not isinstance(xsd_type, tuple) and xsd_type.target_namespace != XSD_NAMESPACE and hasattr(xsd_type, 'primitive_type'):
                yield cast(XsdTypeProtocol, xsd_type)