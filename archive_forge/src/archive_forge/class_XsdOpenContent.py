from typing import cast, Any, Callable, Dict, Iterable, Iterator, List, Optional, \
from elementpath import SchemaElementNode, build_schema_node_tree
from ..exceptions import XMLSchemaValueError
from ..names import XSI_NAMESPACE, XSD_ANY, XSD_ANY_ATTRIBUTE, \
from ..aliases import ElementType, SchemaType, SchemaElementType, SchemaAttributeType, \
from ..translation import gettext as _
from ..helpers import get_namespace, raw_xml_encode
from ..xpath import XsdSchemaProtocol, XsdElementProtocol, XMLSchemaProxy, ElementPathMixin
from .xsdbase import ValidationMixin, XsdComponent
from .particles import ParticleMixin
from . import elements
class XsdOpenContent(XsdComponent):
    """
    Class for XSD 1.1 *openContent* model definitions.

    ..  <openContent
          id = ID
          mode = (none | interleave | suffix) : interleave
          {any attributes with non-schema namespace . . .}>
          Content: (annotation?), (any?)
        </openContent>
    """
    _ADMITTED_TAGS = {XSD_OPEN_CONTENT}
    mode = 'interleave'
    any_element = None

    def __init__(self, elem: ElementType, schema: SchemaType, parent: XsdComponent) -> None:
        super(XsdOpenContent, self).__init__(elem, schema, parent)

    def __repr__(self) -> str:
        return '%s(mode=%r)' % (self.__class__.__name__, self.mode)

    def _parse(self) -> None:
        super(XsdOpenContent, self)._parse()
        try:
            self.mode = self.elem.attrib['mode']
        except KeyError:
            pass
        else:
            if self.mode not in {'none', 'interleave', 'suffix'}:
                msg = _("wrong value %r for 'mode' attribute")
                self.parse_error(msg % self.mode)
        child = self._parse_child_component(self.elem)
        if self.mode == 'none':
            if child is not None and child.tag == XSD_ANY:
                msg = _("an openContent with mode='none' cannot have an <xs:any> child declaration")
                self.parse_error(msg)
        elif child is None or child.tag != XSD_ANY:
            self.parse_error(_('an <xs:any> child declaration is required'))
        else:
            any_element = Xsd11AnyElement(child, self.schema, self)
            any_element.min_occurs = 0
            any_element.max_occurs = None
            self.any_element = any_element

    @property
    def built(self) -> bool:
        return True

    def is_restriction(self, other: 'XsdOpenContent') -> bool:
        if other is None or other.mode == 'none':
            return self.mode == 'none'
        elif self.mode == 'interleave' and other.mode == 'suffix':
            return False
        else:
            return self.any_element.is_restriction(other.any_element)