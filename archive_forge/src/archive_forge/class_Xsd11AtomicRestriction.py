from decimal import DecimalException
from typing import cast, Any, Callable, Dict, Iterator, List, \
from xml.etree import ElementTree
from ..aliases import ElementType, AtomicValueType, ComponentClassType, \
from ..exceptions import XMLSchemaTypeError, XMLSchemaValueError
from ..names import XSD_NAMESPACE, XSD_ANY_TYPE, XSD_SIMPLE_TYPE, XSD_PATTERN, \
from ..translation import gettext as _
from ..helpers import local_name
from .exceptions import XMLSchemaValidationError, XMLSchemaEncodeError, \
from .xsdbase import XsdComponent, XsdType, ValidationMixin
from .facets import XsdFacet, XsdWhiteSpaceFacet, XsdPatternFacets, \
class Xsd11AtomicRestriction(XsdAtomicRestriction):
    """
    Class for XSD 1.1 atomic simpleType and complexType's simpleContent restrictions.

    ..  <restriction
          base = QName
          id = ID
          {any attributes with non-schema namespace . . .}>
          Content: (annotation?, (simpleType?, (minExclusive | minInclusive | maxExclusive |
          maxInclusive | totalDigits | fractionDigits | length | minLength | maxLength |
          enumeration | whiteSpace | pattern | assertion | explicitTimezone |
          {any with namespace: ##other})*))
        </restriction>
    """
    _FACETS_BUILDERS = XSD_11_FACETS_BUILDERS
    _CONTENT_TAIL_TAGS = {XSD_ATTRIBUTE, XSD_ATTRIBUTE_GROUP, XSD_ANY_ATTRIBUTE, XSD_ASSERT}