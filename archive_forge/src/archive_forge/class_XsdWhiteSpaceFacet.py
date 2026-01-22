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
class XsdWhiteSpaceFacet(XsdFacet):
    """
    XSD *whiteSpace* facet.

    ..  <whiteSpace
          fixed = boolean : false
          id = ID
          value = (collapse | preserve | replace)
          {any attributes with non-schema namespace . . .}>
          Content: (annotation?)
        </whiteSpace>
    """
    value: str
    _ADMITTED_TAGS = (XSD_WHITE_SPACE,)

    def _parse_value(self, elem: ElementType) -> None:
        self.value = elem.attrib['value']
        if self.value == 'collapse':
            self._validator = self.collapse_white_space_validator
        elif self.value == 'replace':
            if self.base_value == 'collapse':
                self.parse_error(_("facet value can be only 'collapse'"))
            self._validator = self.replace_white_space_validator
        elif self.base_value == 'collapse':
            self.parse_error(_("facet value can be only 'collapse'"))
        elif self.base_value == 'replace':
            self.parse_error(_("facet value can be only 'replace' or 'collapse'"))

    def replace_white_space_validator(self, value: str) -> None:
        if '\t' in value or '\n' in value:
            raise XMLSchemaValidationError(self, value, _('value contains tabs or newlines'))

    def collapse_white_space_validator(self, value: str) -> None:
        if '\t' in value or '\n' in value or '  ' in value:
            raise XMLSchemaValidationError(self, value, _('value contains non collapsed white spaces'))