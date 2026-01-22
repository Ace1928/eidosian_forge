import re
import math
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Pattern, \
from elementpath import XPath2Parser, ElementPathError, XPathToken, XPathContext, \
from ..exceptions import XMLSchemaTypeError, XMLSchemaValueError
from ..names import XSD_QNAME, XSD_UNIQUE, XSD_KEY, XSD_KEYREF, XSD_SELECTOR, XSD_FIELD
from ..translation import gettext as _
from ..helpers import get_qname, get_extended_qname
from ..aliases import ElementType, SchemaType, NamespacesType, AtomicValueType
from .exceptions import XMLSchemaNotBuiltError
from .xsdbase import XsdComponent
from .attributes import XsdAttribute
from .wildcards import XsdAnyElement
from . import elements
class XsdIdentity(XsdComponent):
    """
    Common class for XSD identity constraints.

    :ivar selector: the XPath selector of the identity constraint.
    :ivar fields: a list containing the XPath field selectors of the identity constraint.
    """
    name: str
    local_name: str
    prefixed_name: str
    parent: 'XsdElement'
    ref: Optional['XsdIdentity']
    selector: Optional[XsdSelector] = None
    fields: Union[Tuple[()], List[XsdFieldSelector]] = ()
    elements: Union[Tuple[()], Dict['XsdElement', Optional[IdentityCounterType]]] = ()
    root_elements: Union[Tuple[()], Set['XsdElement']] = ()

    def __init__(self, elem: ElementType, schema: SchemaType, parent: Optional['XsdElement']) -> None:
        super(XsdIdentity, self).__init__(elem, schema, parent)

    def _parse(self) -> None:
        try:
            self.name = get_qname(self.target_namespace, self.elem.attrib['name'])
        except KeyError:
            self.parse_error(_("missing required attribute 'name'"))
            self.name = ''
        for child in self.elem:
            if child.tag == XSD_SELECTOR:
                self.selector = XsdSelector(child, self.schema, self)
                break
        else:
            self.parse_error(_("missing 'selector' declaration"))
        self.fields = []
        for child in self.elem:
            if child.tag == XSD_FIELD:
                self.fields.append(XsdFieldSelector(child, self.schema, self))

    def build(self) -> None:
        if self.ref is True:
            try:
                ref = self.maps.identities[self.name]
            except KeyError:
                msg = _('unknown identity constraint {!r}')
                self.parse_error(msg.format(self.name))
                return
            else:
                if not isinstance(ref, self.__class__):
                    msg = _("attribute 'ref' points to a different kind constraint")
                    self.parse_error(msg)
                self.selector = ref.selector
                self.fields = ref.fields
                self.ref = ref
        if self.selector is None:
            return
        elif self.selector.token is None:
            raise XMLSchemaNotBuiltError(self, 'identity selector is not built')
        context = XPathContext(self.schema.xpath_node, item=self.parent.xpath_node)
        self.elements = {}
        for e in self.selector.token.select_results(context):
            if not isinstance(e, (elements.XsdElement, XsdAnyElement)):
                msg = _('selector xpath expression can only select elements')
                self.parse_error(msg)
            elif e.name is not None:
                if e.ref is not None:
                    e = e.ref
                self.elements[e] = None
                e.selected_by.add(self)
        if not self.elements:
            qname: Any
            for qname in self.selector.token.iter_leaf_elements():
                xsd_element = self.maps.elements.get(get_extended_qname(qname, self.namespaces))
                if xsd_element is not None and (not isinstance(xsd_element, tuple)) and (xsd_element not in self.elements):
                    if xsd_element.ref is not None:
                        xsd_element = xsd_element.ref
                    self.elements[xsd_element] = None
                    xsd_element.selected_by.add(self)
        self.root_elements = set()
        for token in iter_root_elements(self.selector.token):
            context = XPathContext(self.schema.xpath_node, item=self.parent.xpath_node)
            for e in token.select_results(context):
                if isinstance(e, elements.XsdElement):
                    self.root_elements.add(e)

    @property
    def built(self) -> bool:
        return not isinstance(self.elements, tuple)

    def get_fields(self, element_node: ElementNode, namespaces: Optional[NamespacesType]=None, decoders: Optional[Tuple[XsdAttribute, ...]]=None) -> IdentityCounterType:
        """
        Get fields for a schema or instance context element.

        :param element_node: an Element or an XsdElement
        :param namespaces: is an optional mapping from namespace prefix to URI.
        :param decoders: context schema fields decoders.
        :return: a tuple with field values. An empty field is replaced by `None`.
        """
        fields: List[IdentityFieldItemType] = []

        def append_fields() -> None:
            if isinstance(value, list):
                fields.append(tuple(value))
            elif isinstance(value, bool):
                fields.append((value, bool))
            elif not isinstance(value, float):
                fields.append(value)
            elif math.isnan(value):
                fields.append(('nan', float))
            else:
                fields.append((value, float))
        result: Any
        value: Union[AtomicValueType, None]
        for k, field in enumerate(self.fields):
            if field.token is None:
                msg = f'identity field {field} is not built'
                raise XMLSchemaNotBuiltError(self, msg)
            context = XPathContext(element_node)
            result = field.token.get_results(context)
            if not result:
                if decoders is not None and decoders[k] is not None:
                    value = decoders[k].value_constraint
                    if value is not None:
                        if decoders[k].type.root_type.name == XSD_QNAME:
                            value = get_extended_qname(value, namespaces)
                        append_fields()
                        continue
                if not isinstance(self, XsdKey) or ('ref' in element_node.elem.attrib and self.schema.meta_schema is None and (self.schema.XSD_VERSION != '1.0')):
                    fields.append(None)
                elif field.target_namespace not in self.maps.namespaces:
                    fields.append(None)
                else:
                    msg = _('missing key field {0!r} for {1!r}')
                    raise XMLSchemaValueError(msg.format(field.path, self))
            elif len(result) == 1:
                if decoders is None or decoders[k] is None:
                    fields.append(result[0])
                else:
                    if decoders[k].type.content_type_label not in ('simple', 'mixed'):
                        msg = _("%r field doesn't have a simple type!")
                        raise XMLSchemaTypeError(msg % field)
                    value = decoders[k].data_value(result[0])
                    if decoders[k].type.root_type.name == XSD_QNAME:
                        if isinstance(value, str):
                            value = get_extended_qname(value, namespaces)
                        elif isinstance(value, datatypes.QName):
                            value = value.expanded_name
                    append_fields()
            else:
                msg = _('%r field selects multiple values!')
                raise XMLSchemaValueError(msg % field)
        return tuple(fields)

    def get_counter(self, elem: ElementType) -> 'IdentityCounter':
        return IdentityCounter(self, elem)