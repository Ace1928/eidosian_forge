import warnings
from collections.abc import MutableMapping
from copy import copy as _copy
from typing import TYPE_CHECKING, cast, overload, Any, Iterable, Iterator, \
from xml.etree import ElementTree
from .. import limits
from ..exceptions import XMLSchemaValueError
from ..names import XSD_GROUP, XSD_SEQUENCE, XSD_ALL, XSD_CHOICE, XSD_ELEMENT, \
from ..aliases import ElementType, NamespacesType, SchemaType, IterDecodeType, \
from ..translation import gettext as _
from ..helpers import get_qname, local_name, raw_xml_encode
from ..converters import ElementData
from .exceptions import XMLSchemaModelError, XMLSchemaModelDepthError, \
from .xsdbase import ValidationMixin, XsdComponent, XsdType
from .particles import ParticleMixin, OccursCalculator
from .elements import XsdElement, XsdAlternative
from .wildcards import XsdAnyElement, Xsd11AnyElement
from .models import ModelVisitor, iter_unordered_content, iter_collapsed_content
def _parse_content_model(self, content_model: ElementType) -> None:
    self.model = local_name(content_model.tag)
    if self.model == 'all':
        if self.max_occurs not in (0, 1):
            msg = _("maxOccurs must be (0 | 1) for 'all' model groups")
            self.parse_error(msg)
        if self.min_occurs not in (0, 1):
            msg = _("minOccurs must be (0 | 1) for 'all' model groups")
            self.parse_error(msg)
    for child in content_model:
        if child.tag == XSD_ELEMENT:
            self.append(self.schema.xsd_element_class(child, self.schema, self, False))
        elif child.tag == XSD_ANY:
            self._group.append(Xsd11AnyElement(child, self.schema, self))
        elif child.tag in (XSD_SEQUENCE, XSD_CHOICE, XSD_ALL):
            self._group.append(Xsd11Group(child, self.schema, self))
        elif child.tag == XSD_GROUP:
            try:
                ref = self.schema.resolve_qname(child.attrib['ref'])
            except (KeyError, ValueError, RuntimeError) as err:
                if 'ref' not in child.attrib:
                    msg = _("missing attribute 'ref' in local group")
                    self.parse_error(msg, child)
                else:
                    self.parse_error(err, child)
                continue
            if ref != self.name:
                xsd_group = Xsd11Group(child, self.schema, self)
                self._group.append(xsd_group)
                if (self.model != 'all') ^ (xsd_group.model != 'all'):
                    msg = _('an xs:{0} group cannot include a reference to an xs:{1} group').format(self.model, xsd_group.model)
                    self.parse_error(msg)
                    self.pop()
            elif self.redefine is None:
                msg = _('Circular definition detected for group %r')
                self.parse_error(msg % self.name)
            else:
                if child.get('minOccurs', '1') != '1' or child.get('maxOccurs', '1') != '1':
                    msg = _('Redefined group reference cannot have minOccurs/maxOccurs other than 1')
                    self.parse_error(msg)
                self._group.append(self.redefine)