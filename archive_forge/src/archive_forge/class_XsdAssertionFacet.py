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
class XsdAssertionFacet(XsdFacet):
    """
    XSD 1.1 *assertion* facet for simpleType definitions.

    ..  <assertion
          id = ID
          test = an XPath expression
          xpathDefaultNamespace = (anyURI | (##defaultNamespace | ##targetNamespace | ##local))
          {any attributes with non-schema namespace . . .}>
          Content: (annotation?)
        </assertion>
    """
    _ADMITTED_TAGS = {XSD_ASSERTION}
    _root = ElementNode(elem=Element('root'))

    def __repr__(self) -> str:
        return '%s(test=%r)' % (self.__class__.__name__, self.path)

    def _parse(self) -> None:
        try:
            self.path = self.elem.attrib['test']
        except KeyError:
            self.parse_error(_("missing attribute 'test'"))
            self.path = 'true()'
        try:
            value = self.base_type.primitive_type.prefixed_name
        except AttributeError:
            value = self.any_simple_type.prefixed_name
        if 'xpathDefaultNamespace' in self.elem.attrib:
            self.xpath_default_namespace = self._parse_xpath_default_namespace(self.elem)
        else:
            self.xpath_default_namespace = self.schema.xpath_default_namespace
        self.parser = XsdAssertionXPathParser(namespaces=self.namespaces, strict=False, variable_types={'value': value}, default_namespace=self.xpath_default_namespace)
        try:
            self.token = self.parser.parse(self.path)
        except ElementPathError as err:
            self.parse_error(err)
            self.token = self.parser.parse('true()')

    def __call__(self, value: AtomicValueType) -> None:
        context = XPathContext(self._root, variables={'value': value})
        try:
            if not self.token.evaluate(context):
                reason = _('value is not true with test path {!r}').format(self.path)
                raise XMLSchemaValidationError(self, value, reason)
        except ElementPathError as err:
            raise XMLSchemaValidationError(self, value, reason=str(err)) from None