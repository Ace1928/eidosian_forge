import re
from typing import TYPE_CHECKING, cast, Any, Dict, Generic, List, Iterator, Optional, \
from xml.etree import ElementTree
from elementpath import select
from elementpath.etree import is_etree_element, etree_tostring
from ..exceptions import XMLSchemaValueError, XMLSchemaTypeError
from ..names import XSD_ANNOTATION, XSD_APPINFO, XSD_DOCUMENTATION, \
from ..aliases import ElementType, NamespacesType, SchemaType, BaseXsdType, \
from ..translation import gettext as _
from ..helpers import get_qname, local_name, get_prefixed_qname
from ..resources import XMLResource
from .exceptions import XMLSchemaParseError, XMLSchemaValidationError
def _parse_child_component(self, elem: ElementType, strict: bool=True) -> Optional[ElementType]:
    child = None
    for e in elem:
        if e.tag == XSD_ANNOTATION or callable(e.tag):
            continue
        elif not strict:
            return e
        elif child is not None:
            msg = _('too many XSD components, unexpected {0!r} found at position {1}')
            self.parse_error(msg.format(child, elem[:].index(e)), elem)
            break
        else:
            child = e
    return child