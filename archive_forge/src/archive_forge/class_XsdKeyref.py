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
class XsdKeyref(XsdIdentity):
    """
    Implementation of xs:keyref.

    :ivar refer: reference to a *xs:key* declaration that must be in the same element     or in a descendant element.
    """
    _ADMITTED_TAGS = {XSD_KEYREF}
    refer: Optional[Union[str, XsdKey]] = None
    refer_path = '.'

    def _parse(self) -> None:
        super(XsdKeyref, self)._parse()
        try:
            self.refer = self.schema.resolve_qname(self.elem.attrib['refer'])
        except (KeyError, ValueError, RuntimeError) as err:
            if 'refer' not in self.elem.attrib:
                self.parse_error(_("missing required attribute 'refer'"))
            else:
                self.parse_error(err)

    def build(self) -> None:
        super(XsdKeyref, self).build()
        if isinstance(self.refer, (XsdKey, XsdUnique)):
            return
        elif isinstance(self.ref, XsdKeyref):
            self.refer = self.ref.refer
        if self.refer is None:
            return
        elif isinstance(self.refer, str):
            refer: Optional[XsdIdentity]
            for refer in self.parent.identities:
                if refer.name == self.refer:
                    break
            else:
                refer = None
            if refer is not None and refer.ref is None:
                self.refer = refer
            else:
                try:
                    self.refer = self.maps.identities[self.refer]
                except KeyError:
                    msg = _('key/unique identity constraint %r is missing')
                    self.parse_error(msg % self.refer)
                    return
        if not isinstance(self.refer, (XsdKey, XsdUnique)):
            msg = _('reference to a non key/unique identity constraint %r')
            self.parse_error(msg % self.refer)
        elif len(self.refer.fields) != len(self.fields):
            msg = _('field cardinality mismatch between {0!r} and {1!r}')
            self.parse_error(msg.format(self, self.refer))
        elif self.parent is not self.refer.parent:
            refer_path = self.refer.parent.get_path(ancestor=self.parent)
            if refer_path is None:
                refer_path = self.parent.get_path(ancestor=self.refer.parent, reverse=True)
                if refer_path is None:
                    path1 = self.parent.get_path(reverse=True)
                    path2 = self.refer.parent.get_path()
                    assert path1 is not None
                    assert path2 is not None
                    refer_path = f'{path1}/{path2}'
            self.refer_path = refer_path

    @property
    def built(self) -> bool:
        return not isinstance(self.elements, tuple) and isinstance(self.refer, XsdIdentity)

    def get_counter(self, elem: ElementType) -> 'KeyrefCounter':
        return KeyrefCounter(self, elem)