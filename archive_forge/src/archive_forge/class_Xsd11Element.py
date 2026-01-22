import warnings
from copy import copy as _copy
from decimal import Decimal
from types import GeneratorType
from typing import TYPE_CHECKING, cast, Any, Dict, Iterator, List, Optional, \
from xml.etree.ElementTree import Element
from elementpath import XPath2Parser, ElementPathError, XPathContext, XPathToken, \
from elementpath.datatypes import AbstractDateTime, Duration, AbstractBinary
from ..exceptions import XMLSchemaTypeError, XMLSchemaValueError
from ..names import XSD_COMPLEX_TYPE, XSD_SIMPLE_TYPE, XSD_ALTERNATIVE, \
from ..aliases import ElementType, SchemaType, BaseXsdType, SchemaElementType, \
from ..translation import gettext as _
from ..helpers import get_qname, get_namespace, etree_iter_location_hints, \
from .. import dataobjects
from ..converters import ElementData, XMLSchemaConverter
from ..xpath import XsdSchemaProtocol, XsdElementProtocol, XMLSchemaProxy, \
from ..resources import XMLResource
from .exceptions import XMLSchemaNotBuiltError, XMLSchemaValidationError, \
from .helpers import get_xsd_derivation_attribute
from .xsdbase import XSD_TYPE_DERIVATIONS, XSD_ELEMENT_DERIVATIONS, \
from .particles import ParticleMixin, OccursCalculator
from .identities import XsdIdentity, XsdKey, XsdUnique, \
from .simple_types import XsdSimpleType
from .attributes import XsdAttribute
from .wildcards import XsdAnyElement
class Xsd11Element(XsdElement):
    """
    Class for XSD 1.1 *element* declarations.

    ..  <element
          abstract = boolean : false
          block = (#all | List of (extension | restriction | substitution))
          default = string
          final = (#all | List of (extension | restriction))
          fixed = string
          form = (qualified | unqualified)
          id = ID
          maxOccurs = (nonNegativeInteger | unbounded)  : 1
          minOccurs = nonNegativeInteger : 1
          name = NCName
          nillable = boolean : false
          ref = QName
          substitutionGroup = List of QName
          targetNamespace = anyURI
          type = QName
          {any attributes with non-schema namespace . . .}>
          Content: (annotation?, ((simpleType | complexType)?, alternative*,
          (unique | key | keyref)*))
        </element>
    """
    _target_namespace: Optional[str] = None

    def _parse(self) -> None:
        if not self._build:
            return
        self._parse_particle(self.elem)
        self._parse_attributes()
        if self.ref is None:
            self._parse_type()
            self._parse_alternatives()
            self._parse_constraints()
            if self.parent is None and 'substitutionGroup' in self.elem.attrib:
                for substitution_group in self.elem.attrib['substitutionGroup'].split():
                    self._parse_substitution_group(substitution_group)
        self._parse_target_namespace()
        if any((v.inheritable for v in self.attributes.values())):
            self.inheritable = {}
            for k, v in self.attributes.items():
                if k is not None and isinstance(v, XsdAttribute):
                    if v.inheritable:
                        self.inheritable[k] = v

    def _parse_alternatives(self) -> None:
        alternatives = []
        has_test = True
        for child in self.elem:
            if child.tag == XSD_ALTERNATIVE:
                alternatives.append(XsdAlternative(child, self.schema, self))
                if not has_test:
                    msg = _('test attribute missing in non-final alternative')
                    self.parse_error(msg)
                has_test = 'test' in child.attrib
        if alternatives:
            self.alternatives = alternatives

    @property
    def built(self) -> bool:
        return (self.type.parent is None or self.type.built) and all((c.built for c in self.identities)) and all((a.built for a in self.alternatives))

    @property
    def target_namespace(self) -> str:
        if self._target_namespace is not None:
            return self._target_namespace
        elif self.ref is not None:
            return self.ref.target_namespace
        else:
            return self.schema.target_namespace

    def iter_components(self, xsd_classes: ComponentClassType=None) -> Iterator[XsdComponent]:
        if xsd_classes is None:
            yield self
            yield from self.identities
        else:
            if isinstance(self, xsd_classes):
                yield self
            for obj in self.identities:
                if isinstance(obj, xsd_classes):
                    yield obj
        for alt in self.alternatives:
            yield from alt.iter_components(xsd_classes)
        if self.ref is None and self.type.parent is not None:
            yield from self.type.iter_components(xsd_classes)

    def iter_substitutes(self) -> Iterator[XsdElement]:
        if self.parent is None or self.ref is not None:
            for xsd_element in self.maps.substitution_groups.get(self.name, ()):
                yield xsd_element
                yield from xsd_element.iter_substitutes()

    def get_type(self, elem: Union[ElementType, ElementData], inherited: Optional[Dict[str, Any]]=None) -> BaseXsdType:
        if not self.alternatives:
            return self._head_type or self.type
        if isinstance(elem, ElementData):
            if elem.attributes:
                attrib: Dict[str, str] = {}
                for k, v in elem.attributes.items():
                    value = raw_xml_encode(v)
                    if value is not None:
                        attrib[k] = value
                elem = Element(elem.tag, attrib=attrib)
            else:
                elem = Element(elem.tag)
        if inherited:
            dummy = Element('_dummy_element', attrib=inherited)
            dummy.attrib.update(elem.attrib)
            for alt in self.alternatives:
                if alt.type is not None:
                    if alt.token is None or alt.test(elem) or alt.test(dummy):
                        return alt.type
        else:
            for alt in self.alternatives:
                if alt.type is not None:
                    if alt.token is None or alt.test(elem):
                        return alt.type
        return self._head_type or self.type

    def is_overlap(self, other: SchemaElementType) -> bool:
        if isinstance(other, XsdElement):
            if self.name == other.name:
                return True
            elif any((self.name == x.name for x in other.iter_substitutes())):
                return True
            for e in self.iter_substitutes():
                if other.name == e.name or any((x is e for x in other.iter_substitutes())):
                    return True
        elif isinstance(other, XsdAnyElement):
            if other.is_matching(self.name, self.default_namespace):
                return True
            for e in self.maps.substitution_groups.get(self.name, ()):
                if other.is_matching(e.name, self.default_namespace):
                    return True
        return False

    def is_consistent(self, other: SchemaElementType, strict: bool=True) -> bool:
        if isinstance(other, XsdAnyElement):
            if other.process_contents == 'skip':
                return True
            xsd_element = other.match(self.name, self.default_namespace, resolve=True)
            return xsd_element is None or self.is_consistent(xsd_element, strict=False)
        e1: XsdElement = self
        e2 = other
        if self.name != other.name:
            for e1 in self.iter_substitutes():
                if e1.name == other.name:
                    break
            else:
                for e2 in other.iter_substitutes():
                    if e2.name == self.name:
                        break
                else:
                    return True
        if len(e1.alternatives) != len(e2.alternatives):
            return False
        elif e1.type is not e2.type and strict:
            return False
        elif e1.type is not e2.type or not all((any((a == x for x in e2.alternatives)) for a in e1.alternatives)) or (not all((any((a == x for x in e1.alternatives)) for a in e2.alternatives))):
            msg = _('Maybe a not equivalent type table between elements {0!r} and {1!r}')
            warnings.warn(msg.format(e1, e2), XMLSchemaTypeTableWarning, stacklevel=3)
        return True