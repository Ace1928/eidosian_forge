import re
import math
import operator
from abc import abstractmethod
from typing import TYPE_CHECKING, cast, Any, List, Optional, Pattern, Union, \
from xml.etree.ElementTree import Element
from elementpath import XPath2Parser, XPathContext, ElementPathError, \
from ..names import XSD_LENGTH, XSD_MIN_LENGTH, XSD_MAX_LENGTH, XSD_ENUMERATION, \
from ..aliases import ElementType, SchemaType, AtomicValueType, BaseXsdType
from ..translation import gettext as _
from ..helpers import count_digits, local_name
from .exceptions import XMLSchemaValidationError, XMLSchemaDecodeError
from .xsdbase import XsdComponent, XsdAnnotation
class XsdFractionDigitsFacet(XsdFacet):
    """
    XSD *fractionDigits* facet.

    ..  <fractionDigits
          fixed = boolean : false
          id = ID
          value = nonNegativeInteger
          {any attributes with non-schema namespace . . .}>
          Content: (annotation?)
        </fractionDigits>
    """
    value: int
    base_type: BaseXsdType
    _ADMITTED_TAGS = (XSD_FRACTION_DIGITS,)

    def __init__(self, elem: ElementType, schema: SchemaType, parent: 'XsdAtomicRestriction', base_type: BaseXsdType) -> None:
        super(XsdFractionDigitsFacet, self).__init__(elem, schema, parent, base_type)
        if not base_type.is_derived(self.maps.types[XSD_DECIMAL]):
            msg = _('fractionDigits facet can be applied only to types derived from xs:decimal')
            self.parse_error(msg)

    def _parse_value(self, elem: ElementType) -> None:
        try:
            self.value = int(elem.attrib['value'])
        except (ValueError, KeyError):
            self.value = 9999
        else:
            if self.value < 0:
                self.value = 9999
            elif self.value > 0 and self.base_type.is_derived(self.maps.types[XSD_INTEGER]):
                msg = _('fractionDigits facet value must be 0 for types derived from xs:integer')
                raise ValueError(msg)
            facet: Any = self.base_type.get_facet(XSD_FRACTION_DIGITS)
            if facet is not None and facet.value < self.value:
                msg = _('invalid restriction: base value is lower ({})')
                self.parse_error(msg.format(facet.value))

    def __call__(self, value: Any) -> None:
        try:
            if count_digits(value)[1] <= self.value:
                return
        except (TypeError, ValueError, ArithmeticError) as err:
            raise XMLSchemaValidationError(self, value, str(err)) from None
        else:
            reason = _('the number of fraction digits has to be lesser or equal than {!r}').format(self.value)
            raise XMLSchemaValidationError(self, value, reason)