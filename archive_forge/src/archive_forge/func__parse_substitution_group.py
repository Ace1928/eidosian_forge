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
def _parse_substitution_group(self, substitution_group: str) -> None:
    try:
        substitution_group_qname = self.schema.resolve_qname(substitution_group)
    except (KeyError, ValueError, RuntimeError) as err:
        self.parse_error(err)
        return
    else:
        if substitution_group_qname[0] != '{':
            substitution_group_qname = get_qname(self.target_namespace, substitution_group_qname)
    try:
        head_element = self.maps.lookup_element(substitution_group_qname)
    except KeyError:
        msg = _('unknown substitutionGroup %r')
        self.parse_error(msg % substitution_group)
        return
    else:
        if isinstance(head_element, tuple):
            msg = _('circularity found for substitutionGroup %r')
            self.parse_error(msg % substitution_group)
            return
        elif 'substitution' in head_element.block:
            return
    final = head_element.final
    if self.type == head_element.type:
        pass
    elif self.type.name == XSD_ANY_TYPE:
        if head_element.type.name != XSD_ANY_TYPE:
            self._head_type = head_element.type
    elif not self.type.is_derived(head_element.type):
        msg = _('{0!r} type is not of the same or a derivation of the head element {1!r} type')
        self.parse_error(msg.format(self, head_element))
    elif final == '#all' or ('extension' in final and 'restriction' in final):
        msg = _("head element %r can't be substituted by an element that has a derivation of its type")
        self.parse_error(msg % head_element)
    elif 'extension' in final and self.type.is_derived(head_element.type, 'extension'):
        msg = _("head element %r can't be substituted by an element that has an extension of its type")
        self.parse_error(msg % head_element)
    elif 'restriction' in final and self.type.is_derived(head_element.type, 'restriction'):
        msg = _("head element %r can't be substituted by an element that has a restriction of its type")
        self.parse_error(msg % head_element)
    try:
        self.maps.substitution_groups[substitution_group_qname].add(self)
    except KeyError:
        self.maps.substitution_groups[substitution_group_qname] = {self}
    finally:
        self.substitution_group = substitution_group_qname