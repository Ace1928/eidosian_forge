from typing import cast, Any, Callable, Dict, Iterable, Iterator, List, Optional, \
from elementpath import SchemaElementNode, build_schema_node_tree
from ..exceptions import XMLSchemaValueError
from ..names import XSI_NAMESPACE, XSD_ANY, XSD_ANY_ATTRIBUTE, \
from ..aliases import ElementType, SchemaType, SchemaElementType, SchemaAttributeType, \
from ..translation import gettext as _
from ..helpers import get_namespace, raw_xml_encode
from ..xpath import XsdSchemaProtocol, XsdElementProtocol, XMLSchemaProxy, ElementPathMixin
from .xsdbase import ValidationMixin, XsdComponent
from .particles import ParticleMixin
from . import elements
def _parse_not_constraints(self) -> None:
    if 'notNamespace' not in self.elem.attrib:
        pass
    elif 'namespace' in self.elem.attrib:
        msg = _("'namespace' and 'notNamespace' attributes are mutually exclusive")
        self.parse_error(msg)
    else:
        self.namespace = []
        self.not_namespace = []
        for ns in self.elem.attrib['notNamespace'].strip().split():
            if ns == '##local':
                self.not_namespace.append('')
            elif ns == '##targetNamespace':
                self.not_namespace.append(self.target_namespace)
            elif ns.startswith('##'):
                msg = _("wrong value %r in 'notNamespace' attribute")
                self.parse_error(msg % ns)
            else:
                self.not_namespace.append(ns)
    if 'notQName' not in self.elem.attrib:
        return
    not_qname = self.elem.attrib['notQName'].strip().split()
    if isinstance(self, XsdAnyAttribute) and (not all((not s.startswith('##') or s == '##defined' for s in not_qname))) or not all((not s.startswith('##') or s in {'##defined', '##definedSibling'} for s in not_qname)):
        self.parse_error(_("wrong value for 'notQName' attribute"))
        return
    try:
        names = [x if x.startswith('##') else self.schema.resolve_qname(x, False) for x in not_qname]
    except KeyError as err:
        msg = _("unmapped QName in 'notQName' attribute: %s")
        self.parse_error(msg % str(err))
        return
    except ValueError as err:
        msg = _("wrong QName format in 'notQName' attribute: %s")
        self.parse_error(msg % str(err))
        return
    if self.not_namespace:
        if any((not x.startswith('##') for x in names)) and all((get_namespace(x) in self.not_namespace for x in names if not x.startswith('##'))):
            msg = _('the namespace of each QName in notQName is allowed by notNamespace')
            self.parse_error(msg)
    elif any((not self.is_namespace_allowed(get_namespace(x)) for x in names if not x.startswith('##'))):
        msg = _('names in notQName must be in namespaces that are allowed')
        self.parse_error(msg)
    self.not_qname = names