from typing import cast, Any, Callable, Iterator, List, Optional, Tuple, Union
from elementpath.datatypes import AnyAtomicType
from ..exceptions import XMLSchemaValueError
from ..names import XSD_GROUP, XSD_ATTRIBUTE_GROUP, XSD_SEQUENCE, XSD_OVERRIDE, \
from ..aliases import ElementType, NamespacesType, SchemaType, ComponentClassType, \
from ..translation import gettext as _
from ..helpers import get_qname, local_name
from .exceptions import XMLSchemaDecodeError
from .helpers import get_xsd_derivation_attribute
from .xsdbase import XSD_TYPE_DERIVATIONS, XsdComponent, XsdType, ValidationMixin
from .attributes import XsdAttributeGroup
from .assertions import XsdAssert
from .simple_types import FacetsValueType, XsdSimpleType, XsdUnion
from .groups import XsdGroup
from .wildcards import XsdOpenContent, XsdDefaultOpenContent
def _parse_derivation_elem(self, elem: ElementType) -> Optional[ElementType]:
    derivation_elem = self._parse_child_component(elem)
    if derivation_elem is None or derivation_elem.tag not in {XSD_RESTRICTION, XSD_EXTENSION}:
        msg = _('restriction or extension tag expected')
        self.parse_error(msg, derivation_elem)
        self.content = self.schema.create_any_content_group(self)
        self.attributes = self.schema.create_any_attribute_group(self)
        return None
    if self.derivation is not None and self.redefine is None:
        msg = _('{!r} is expected to have a redefined/overridden component')
        raise XMLSchemaValueError(msg.format(self))
    self.derivation = local_name(derivation_elem.tag)
    if self.base_type is not None and self.derivation in self.base_type.final:
        msg = _('{0!r} derivation not allowed for {1!r}')
        self.parse_error(msg.format(self.derivation, self))
    return derivation_elem