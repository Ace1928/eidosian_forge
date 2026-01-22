from decimal import DecimalException
from typing import cast, Any, Callable, Dict, Iterator, List, \
from xml.etree import ElementTree
from ..aliases import ElementType, AtomicValueType, ComponentClassType, \
from ..exceptions import XMLSchemaTypeError, XMLSchemaValueError
from ..names import XSD_NAMESPACE, XSD_ANY_TYPE, XSD_SIMPLE_TYPE, XSD_PATTERN, \
from ..translation import gettext as _
from ..helpers import local_name
from .exceptions import XMLSchemaValidationError, XMLSchemaEncodeError, \
from .xsdbase import XsdComponent, XsdType, ValidationMixin
from .facets import XsdFacet, XsdWhiteSpaceFacet, XsdPatternFacets, \
class XsdAtomic(XsdSimpleType):
    """
    Class for atomic simpleType definitions. An atomic definition has
    a base_type attribute that refers to primitive or derived atomic
    built-in type or another derived simpleType.
    """
    _special_types = {XSD_ANY_TYPE, XSD_ANY_SIMPLE_TYPE, XSD_ANY_ATOMIC_TYPE}
    _ADMITTED_TAGS = {XSD_RESTRICTION, XSD_SIMPLE_TYPE}

    def __init__(self, elem: ElementType, schema: SchemaType, parent: Optional[XsdComponent]=None, name: Optional[str]=None, facets: Optional[Dict[Optional[str], FacetsValueType]]=None, base_type: Optional[BaseXsdType]=None) -> None:
        if base_type is None:
            self.primitive_type = self
        else:
            self.base_type = base_type
        super(XsdAtomic, self).__init__(elem, schema, parent, name, facets)

    def __repr__(self) -> str:
        if self.name is None:
            return '%s(primitive_type=%r)' % (self.__class__.__name__, self.primitive_type.local_name)
        else:
            return '%s(name=%r)' % (self.__class__.__name__, self.prefixed_name)

    def __setattr__(self, name: str, value: Any) -> None:
        super(XsdAtomic, self).__setattr__(name, value)
        if name == 'base_type':
            if not hasattr(self, 'white_space'):
                try:
                    self.white_space = value.white_space
                except AttributeError:
                    pass
            try:
                if value.is_simple():
                    self.primitive_type = value.primitive_type
                else:
                    self.primitive_type = value.content.primitive_type
            except AttributeError:
                self.primitive_type = value

    @property
    def variety(self) -> Optional[str]:
        return 'atomic'

    @property
    def admitted_facets(self) -> Set[str]:
        if self.primitive_type.is_complex():
            return XSD_10_FACETS if self.xsd_version == '1.0' else XSD_11_FACETS
        return self.primitive_type.admitted_facets

    def is_datetime(self) -> bool:
        return self.primitive_type.to_python.__name__ == 'fromstring'

    def get_facet(self, tag: str) -> Optional[FacetsValueType]:
        facet = self.facets.get(tag)
        if facet is not None:
            return facet
        elif self.base_type is not None:
            return self.base_type.get_facet(tag)
        else:
            return None

    def is_atomic(self) -> bool:
        return True