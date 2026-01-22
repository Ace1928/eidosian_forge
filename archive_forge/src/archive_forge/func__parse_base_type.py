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
def _parse_base_type(self, elem: ElementType, complex_content: bool=False) -> Union[XsdSimpleType, 'XsdComplexType']:
    try:
        base_qname = self.schema.resolve_qname(elem.attrib['base'])
    except (KeyError, ValueError, RuntimeError) as err:
        if 'base' not in elem.attrib:
            msg = _("'base' attribute required")
            self.parse_error(msg, elem)
        else:
            self.parse_error(err, elem)
        return self.any_type
    try:
        base_type = self.maps.lookup_type(base_qname)
    except KeyError:
        msg = _('missing base type %r')
        self.parse_error(msg % base_qname, elem)
        if complex_content:
            return self.any_type
        else:
            return self.any_simple_type
    else:
        if isinstance(base_type, tuple):
            msg = _('circular definition found between {0!r} and {1!r}')
            self.parse_error(msg.format(self, base_qname), elem)
            return self.any_type
        elif complex_content and base_type.is_simple():
            msg = _('a complexType ancestor required: {!r}')
            self.parse_error(msg.format(base_type), elem)
            return self.any_type
        if base_type.final and elem.tag.rsplit('}', 1)[-1] in base_type.final:
            msg = _("derivation by %r blocked by attribute 'final' in base type")
            self.parse_error(msg % elem.tag.rsplit('}', 1)[-1])
        return base_type