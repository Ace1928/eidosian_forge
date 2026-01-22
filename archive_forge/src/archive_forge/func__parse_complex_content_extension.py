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
def _parse_complex_content_extension(self, elem: ElementType, base_type: Any) -> None:
    if base_type.is_simple() or base_type.has_simple_content():
        msg = _('base %r is simple or has a simple content')
        self.parse_error(msg % base_type, elem)
        base_type = self.any_type
    if 'extension' in base_type.final:
        msg = _('the base type is not derivable by extension')
        self.parse_error(msg)
    group_elem: Any
    for group_elem in elem:
        if group_elem.tag == XSD_ANNOTATION or callable(group_elem.tag):
            continue
        elif group_elem.tag != XSD_OPEN_CONTENT:
            break
        self.open_content = XsdOpenContent(group_elem, self.schema, self)
        try:
            any_element = base_type.open_content.any_element
            self.open_content.any_element.union(any_element)
        except AttributeError:
            pass
    else:
        group_elem = None
    if not base_type.content:
        if not base_type.mixed:
            if group_elem is not None and group_elem.tag in XSD_MODEL_GROUP_TAGS:
                self.content = self.schema.xsd_group_class(group_elem, self.schema, self)
            else:
                max_occurs = base_type.content.max_occurs
                self.content = self.schema.create_empty_content_group(parent=self, model=base_type.content.model, minOccurs=str(base_type.content.min_occurs), maxOccurs='unbounded' if max_occurs is None else str(max_occurs))
        else:
            self.content = self.schema.create_empty_content_group(self)
            self.content.append(self.schema.create_empty_content_group(self.content))
            if group_elem is not None and group_elem.tag in XSD_MODEL_GROUP_TAGS:
                group = self.schema.xsd_group_class(group_elem, self.schema, self.content)
                if not self.mixed:
                    msg = _('base has a different content type (mixed=%r) and the extension group is not empty.')
                    self.parse_error(msg % base_type.mixed, elem)
                if group.model == 'all':
                    msg = _('cannot extend an empty mixed content with an xs:all')
                    self.parse_error(msg)
            else:
                group = self.schema.create_empty_content_group(self)
            self.content.append(group)
            self.content.elem.append(base_type.content.elem)
            self.content.elem.append(group.elem)
    elif group_elem is not None and group_elem.tag in XSD_MODEL_GROUP_TAGS:
        group = self.schema.xsd_group_class(group_elem, self.schema, self)
        if base_type.content.model != 'all':
            content = self.schema.create_empty_content_group(self)
            content.append(base_type.content)
            content.elem.append(base_type.content.elem)
            if group.model == 'all':
                msg = _('xs:all cannot extend a not empty xs:%s')
                self.parse_error(msg % base_type.content.model)
            else:
                content.append(group)
                content.elem.append(group.elem)
        else:
            content = self.schema.create_empty_content_group(self, model='all', minOccurs=str(base_type.content.min_occurs))
            content.extend(base_type.content)
            content.elem.extend(base_type.content.elem)
            if not group:
                pass
            elif group.model != 'all':
                msg = _("cannot extend a not empty 'all' model group with a different model")
                self.parse_error(msg)
            elif base_type.content.min_occurs != group.min_occurs:
                msg = _('when extend an xs:all group minOccurs must be the same')
                self.parse_error(msg)
            elif base_type.mixed and (not base_type.content):
                msg = _('cannot extend an xs:all group with mixed empty content')
                self.parse_error(msg)
            else:
                content.extend(group)
                content.elem.extend(group.elem)
        if base_type.mixed is not self.mixed:
            msg = _('base has a different content type (mixed=%r) and the extension group is not empty.')
            self.parse_error(msg % base_type.mixed, elem)
        self.content = content
    elif base_type.is_simple():
        self.content = base_type
    elif base_type.has_simple_content():
        self.content = base_type.content
    else:
        if self.mixed is not base_type.mixed:
            if self.mixed:
                msg = _('extended type has a mixed content but the base is element-only')
                self.parse_error(msg, elem)
            self.mixed = base_type.mixed
        self.content = self.schema.create_empty_content_group(self)
        self.content.append(base_type.content)
        self.content.elem.append(base_type.content.elem)
    if self.open_content is None:
        default_open_content = self.default_open_content
        if default_open_content is not None and (self.mixed or self.content or default_open_content.applies_to_empty):
            self.open_content = default_open_content
        elif base_type.open_content is not None:
            self.open_content = base_type.open_content
    if base_type.open_content is not None and self.open_content is not None and (self.open_content is not base_type.open_content):
        if self.open_content.mode == 'none':
            self.open_content = base_type.open_content
        elif not base_type.open_content.is_restriction(self.open_content):
            msg = _('{0!r} is not an extension of the base type {1!r}')
            self.parse_error(msg.format(self.open_content, base_type.open_content))
    self._parse_content_tail(elem, derivation='extension', base_attributes=base_type.attributes)