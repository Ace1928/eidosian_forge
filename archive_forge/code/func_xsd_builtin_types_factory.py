from decimal import Decimal
from elementpath import datatypes
from typing import cast, Any, Dict, Optional, Type, Tuple, Union
from xml.etree.ElementTree import Element
from ..exceptions import XMLSchemaValueError
from ..names import XSD_LENGTH, XSD_MIN_LENGTH, XSD_MAX_LENGTH, XSD_ENUMERATION, \
from ..aliases import ElementType, SchemaType, BaseXsdType
from .helpers import decimal_validator, qname_validator, byte_validator, \
from .facets import XSD_10_FACETS_BUILDERS, XSD_11_FACETS_BUILDERS
from .simple_types import XsdSimpleType, XsdAtomicBuiltin
def xsd_builtin_types_factory(meta_schema: SchemaType, xsd_types: Dict[str, Union[BaseXsdType, Tuple[ElementType, SchemaType]]], atomic_builtin_class: Optional[Type[XsdAtomicBuiltin]]=None) -> None:
    """
    Builds the dictionary for XML Schema built-in types mapping.
    """
    atomic_builtin_class = atomic_builtin_class or XsdAtomicBuiltin
    if meta_schema.XSD_VERSION == '1.1':
        builtin_types = XSD_11_BUILTIN_TYPES
        facets_map = XSD_11_FACETS_BUILDERS
    else:
        builtin_types = XSD_10_BUILTIN_TYPES
        facets_map = XSD_10_FACETS_BUILDERS
    xsd_types[XSD_ANY_TYPE] = meta_schema.create_any_type()
    xsd_any_simple_type = xsd_types[XSD_ANY_SIMPLE_TYPE] = XsdSimpleType(elem=Element(XSD_SIMPLE_TYPE, name=XSD_ANY_SIMPLE_TYPE), schema=meta_schema, parent=None, name=XSD_ANY_SIMPLE_TYPE)
    xsd_types[XSD_ANY_ATOMIC_TYPE] = meta_schema.xsd_atomic_restriction_class(elem=Element(XSD_SIMPLE_TYPE, name=XSD_ANY_ATOMIC_TYPE), schema=meta_schema, parent=None, name=XSD_ANY_ATOMIC_TYPE, base_type=xsd_any_simple_type)
    for item in builtin_types:
        item = item.copy()
        name: str = item['name']
        try:
            value = cast(Tuple[ElementType, SchemaType], xsd_types[name])
        except KeyError:
            elem = Element(XSD_SIMPLE_TYPE, name=name, id=name)
        else:
            elem, schema = value
            if schema is not meta_schema:
                raise XMLSchemaValueError('loaded entry schema is not the meta-schema!')
        base_type: Union[None, BaseXsdType, Tuple[ElementType, SchemaType]]
        if 'base_type' in item:
            base_type = item['base_type'] = xsd_types[item['base_type']]
        else:
            base_type = None
        facets = item.pop('facets', None)
        builtin_type: XsdAtomicBuiltin = atomic_builtin_class(elem, meta_schema, **item)
        if facets:
            built_facets = builtin_type.facets
            for e in facets:
                try:
                    cls: Any = facets_map[e.tag]
                except AttributeError:
                    built_facets[None] = e
                else:
                    built_facets[e.tag] = cls(e, meta_schema, builtin_type, base_type)
            builtin_type.facets = built_facets
        xsd_types[name] = builtin_type