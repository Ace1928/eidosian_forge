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
class Xsd11ComplexType(XsdComplexType):
    """
    Class for XSD 1.1 *complexType* definitions.

    ..  <complexType
          abstract = boolean : false
          block = (#all | List of (extension | restriction))
          final = (#all | List of (extension | restriction))
          id = ID
          mixed = boolean
          name = NCName
          defaultAttributesApply = boolean : true
          {any attributes with non-schema namespace . . .}>
          Content: (annotation?, (simpleContent | complexContent | (openContent?,
          (group | all | choice | sequence)?,
          ((attribute | attributeGroup)*, anyAttribute?), assert*)))
        </complexType>
    """
    default_attributes_apply = True
    _CONTENT_TAIL_TAGS = {XSD_ATTRIBUTE_GROUP, XSD_ATTRIBUTE, XSD_ANY_ATTRIBUTE, XSD_ASSERT}

    @property
    def default_attributes(self) -> Optional[XsdAttributeGroup]:
        if self.redefine is not None:
            default_attributes = self.schema.default_attributes
        else:
            for child in self.schema.root:
                if child.tag == XSD_OVERRIDE and self.elem in child:
                    schema = self.schema.includes[child.attrib['schemaLocation']]
                    if schema.override is self.schema:
                        default_attributes = schema.default_attributes
                        break
            else:
                default_attributes = self.schema.default_attributes
        if isinstance(default_attributes, str):
            return None
        return default_attributes

    @property
    def default_open_content(self) -> Optional[XsdDefaultOpenContent]:
        if self.parent is not None:
            return self.schema.default_open_content
        for child in self.schema.root:
            if child.tag == XSD_OVERRIDE and self.elem in child:
                schema = self.schema.includes[child.attrib['schemaLocation']]
                if schema.override is self.schema:
                    return schema.default_open_content
        else:
            return self.schema.default_open_content

    def _parse(self) -> None:
        super(Xsd11ComplexType, self)._parse()
        if self.base_type and self.base_type.base_type is self.any_simple_type and (self.base_type.derivation == 'extension') and (not self.attributes):
            msg = _('the simple content of {!r} is not a valid simple type in XSD 1.1')
            self.parse_error(msg.format(self.base_type))
        if isinstance(self.content, XsdGroup):
            if self.open_content is None:
                if self.content.interleave is not None or self.content.suffix is not None:
                    msg = _('openContent mismatch between type and model group')
                    self.parse_error(msg)
            elif self.open_content.mode == 'interleave':
                self.content.interleave = self.content.suffix = self.open_content.any_element
            elif self.open_content.mode == 'suffix':
                self.content.suffix = self.open_content.any_element
        if isinstance(self.base_type, XsdComplexType):
            for name, attr in self.base_type.attributes.items():
                if attr.inheritable:
                    if name not in self.attributes:
                        self.attributes[name] = attr
                    elif not self.attributes[name].inheritable:
                        msg = _('attribute %r must be inheritable')
                        self.parse_error(msg % name)
        if 'defaultAttributesApply' not in self.elem.attrib:
            self.default_attributes_apply = True
        elif self.elem.attrib['defaultAttributesApply'].strip() in {'false', '0'}:
            self.default_attributes_apply = False
        else:
            self.default_attributes_apply = True
        if self.default_attributes_apply and isinstance(self.default_attributes, XsdAttributeGroup):
            if self.redefine is None:
                for k in self.default_attributes:
                    if k in self.attributes:
                        msg = _('default attribute {!r} is already declared in the complex type')
                        self.parse_error(msg.format(k))
            self.attributes.update(((k, v) for k, v in self.default_attributes.items()))

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

    def _parse_content_tail(self, elem: ElementType, **kwargs: Any) -> None:
        self.attributes = self.schema.xsd_attribute_group_class(elem, self.schema, self, **kwargs)
        self.assertions = [XsdAssert(e, self.schema, self, self) for e in elem if e.tag == XSD_ASSERT]
        if isinstance(self.base_type, XsdComplexType):
            self.assertions.extend((XsdAssert(assertion.elem, self.schema, self, self) for assertion in self.base_type.assertions))