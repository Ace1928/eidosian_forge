from copy import copy as _copy
from decimal import Decimal
from elementpath.datatypes import AbstractDateTime, Duration, AbstractBinary
from typing import cast, Any, Callable, Union, Dict, List, Optional, \
from ..exceptions import XMLSchemaValueError
from ..names import XSI_NAMESPACE, XSD_ANY_SIMPLE_TYPE, XSD_SIMPLE_TYPE, \
from ..aliases import ComponentClassType, ElementType, IterDecodeType, \
from ..translation import gettext as _
from ..helpers import get_namespace, get_qname
from .exceptions import XMLSchemaValidationError
from .xsdbase import XsdComponent, XsdAnnotation, ValidationMixin
from .simple_types import XsdSimpleType
from .wildcards import XsdAnyAttribute
class Xsd11Attribute(XsdAttribute):
    """
    Class for XSD 1.1 *attribute* declarations.

    ..  <attribute
          default = string
          fixed = string
          form = (qualified | unqualified)
          id = ID
          name = NCName
          ref = QName
          targetNamespace = anyURI
          type = QName
          use = (optional | prohibited | required) : optional
          inheritable = boolean
          {any attributes with non-schema namespace . . .}>
          Content: (annotation?, simpleType?)
        </attribute>
    """
    _target_namespace: Optional[str] = None

    @property
    def target_namespace(self) -> str:
        if self._target_namespace is not None:
            return self._target_namespace
        elif self.ref is not None:
            return self.ref.target_namespace
        else:
            return self.schema.target_namespace

    def _parse(self) -> None:
        super()._parse()
        if self.use == 'prohibited' and 'fixed' in self.elem.attrib:
            msg = _("attribute 'fixed' with use=prohibited is not allowed in XSD 1.1")
            self.parse_error(msg)
        if 'inheritable' in self.elem.attrib:
            if self.elem.attrib['inheritable'].strip() in {'true', '1'}:
                self.inheritable = True
        self._parse_target_namespace()