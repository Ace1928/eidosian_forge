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
class Xsd11AnyElement(XsdAnyElement):
    """
    Class for XSD 1.1 *any* declarations.

    ..  <any
          id = ID
          maxOccurs = (nonNegativeInteger | unbounded)  : 1
          minOccurs = nonNegativeInteger : 1
          namespace = ((##any | ##other) | List of (anyURI | (##targetNamespace | ##local)) )
          notNamespace = List of (anyURI | (##targetNamespace | ##local))
          notQName = List of (QName | (##defined | ##definedSibling))
          processContents = (lax | skip | strict) : strict
          {any attributes with non-schema namespace . . .}>
          Content: (annotation?)
        </any>
    """

    def _parse(self) -> None:
        super(Xsd11AnyElement, self)._parse()
        self._parse_not_constraints()

    def is_matching(self, name: Optional[str], default_namespace: Optional[str]=None, group: Optional[ModelGroupType]=None, occurs: Optional[OccursCounterType]=None, **kwargs: Any) -> bool:
        """
        Returns `True` if the component name is matching the name provided as argument,
        `False` otherwise.

        :param name: a local or fully-qualified name.
        :param default_namespace: used by the XPath processor for completing         the name argument in case it's a local name.
        :param group: used only by XSD 1.1 any element wildcards to verify siblings in         case of ##definedSibling value in notQName attribute.
        :param occurs: a Counter instance for verify model occurrences counting.
        """
        if name is None:
            return False
        elif not name or name[0] == '{':
            if not self.is_namespace_allowed(get_namespace(name)):
                return False
        elif not default_namespace:
            if not self.is_namespace_allowed(''):
                return False
        else:
            name = f'{{{default_namespace}}}{name}'
            if not self.is_namespace_allowed(default_namespace):
                return False
        if group in self.precedences:
            if occurs is None:
                if any((e.is_matching(name) for e in self.precedences[group])):
                    return False
            elif any((e.is_matching(name) and (not e.is_over(occurs[e])) for e in self.precedences[group])):
                return False
        if '##defined' in self.not_qname and name in self.maps.elements:
            return False
        if group and '##definedSibling' in self.not_qname:
            if any((e.is_matching(name) for e in group.iter_elements() if not isinstance(e, XsdAnyElement))):
                return False
        return name not in self.not_qname

    def is_consistent(self, other: SchemaElementType, **kwargs: Any) -> bool:
        if isinstance(other, XsdAnyElement) or self.process_contents == 'skip':
            return True
        xsd_element = self.match(other.name, other.default_namespace, resolve=True)
        return xsd_element is None or other.is_consistent(xsd_element, strict=False)

    def add_precedence(self, other: ModelParticleType, group: ModelGroupType) -> None:
        try:
            self.precedences[group].append(other)
        except KeyError:
            self.precedences[group] = [other]