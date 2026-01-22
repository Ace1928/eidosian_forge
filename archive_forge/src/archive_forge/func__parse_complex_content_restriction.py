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
def _parse_complex_content_restriction(self, elem: ElementType, base_type: Any) -> None:
    if 'restriction' in base_type.final:
        msg = _('the base type is not derivable by restriction')
        self.parse_error(msg)
    if base_type.is_simple() or base_type.has_simple_content():
        msg = _('base %r is simple or has a simple content')
        self.parse_error(msg % base_type, elem)
        base_type = self.any_type
    for child in elem:
        if child.tag == XSD_OPEN_CONTENT and self.xsd_version > '1.0':
            self.open_content = XsdOpenContent(child, self.schema, self)
            continue
        elif child.tag in XSD_MODEL_GROUP_TAGS:
            content = self.schema.xsd_group_class(child, self.schema, self)
            if not base_type.content.admits_restriction(content.model):
                msg = _('restriction of an xs:{0} with more than one particle with xs:{1} is forbidden')
                self.parse_error(msg.format(base_type.content.model, content.model))
            break
    else:
        content = self.schema.create_empty_content_group(self, base_type.content.model)
    content.restriction = base_type.content
    if base_type.is_element_only() and content.mixed:
        msg = _('derived a mixed content from a base type that has element-only content')
        self.parse_error(msg, elem)
    elif base_type.is_empty() and (not content.is_empty()):
        msg = _('an empty content derivation from base type that has not empty content')
        self.parse_error(msg, elem)
    if self.open_content is None:
        default_open_content = self.default_open_content
        if default_open_content is not None and (self.mixed or content or default_open_content.applies_to_empty):
            self.open_content = default_open_content
    if self.open_content and content and (not self.open_content.is_restriction(base_type.open_content)):
        msg = _('{0!r} is not a restriction of the base type {1!r}')
        self.parse_error(msg.format(self.open_content, base_type.open_content))
    self.content = content
    self._parse_content_tail(elem, derivation='restriction', base_attributes=base_type.attributes)