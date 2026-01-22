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
class XsdWildcard(XsdComponent):
    names = ()
    namespace: Union[Tuple[str], List[str]] = ('##any',)
    not_namespace: Union[Tuple[()], List[str]] = ()
    not_qname: Union[Tuple[()], List[str]] = ()
    process_contents = 'strict'
    type = None
    default = None
    fixed = None

    def __repr__(self) -> str:
        if self.not_namespace:
            return '%s(not_namespace=%r, process_contents=%r)' % (self.__class__.__name__, self.not_namespace, self.process_contents)
        else:
            return '%s(namespace=%r, process_contents=%r)' % (self.__class__.__name__, self.namespace, self.process_contents)

    def _parse(self) -> None:
        namespace = self.elem.attrib.get('namespace', '##any').strip()
        if namespace == '##any':
            pass
        elif not namespace:
            self.namespace = []
        elif namespace == '##other':
            self.namespace = [namespace]
        elif namespace == '##local':
            self.namespace = ['']
        elif namespace == '##targetNamespace':
            self.namespace = [self.target_namespace]
        else:
            self.namespace = []
            for ns in namespace.split():
                if ns == '##local':
                    self.namespace.append('')
                elif ns == '##targetNamespace':
                    self.namespace.append(self.target_namespace)
                elif ns.startswith('##'):
                    msg = _("wrong value %r in 'namespace' attribute")
                    self.parse_error(msg % ns)
                else:
                    self.namespace.append(ns)
        process_contents = self.elem.attrib.get('processContents', 'strict')
        if process_contents == 'strict':
            pass
        elif process_contents not in ('lax', 'skip'):
            msg = _("wrong value %r for 'processContents' attribute")
            self.parse_error(msg % self.process_contents)
        else:
            self.process_contents = process_contents

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

    @property
    def built(self) -> bool:
        return True

    @property
    def value_constraint(self) -> Optional[str]:
        return None

    def is_matching(self, name: Optional[str], default_namespace: Optional[str]=None, **kwargs: Any) -> bool:
        if name is None:
            return False
        elif not name or name[0] == '{':
            return self.is_namespace_allowed(get_namespace(name))
        elif not default_namespace:
            return self.is_namespace_allowed('')
        else:
            return self.is_namespace_allowed(default_namespace)

    def is_namespace_allowed(self, namespace: str) -> bool:
        if self.not_namespace:
            return namespace not in self.not_namespace
        elif '##any' in self.namespace or namespace == XSI_NAMESPACE:
            return True
        elif '##other' in self.namespace:
            if not namespace:
                return False
            return namespace != self.target_namespace
        else:
            return namespace in self.namespace

    def deny_namespaces(self, namespaces: List[str]) -> bool:
        if self.not_namespace:
            return all((x in self.not_namespace for x in namespaces))
        elif '##any' in self.namespace:
            return False
        elif '##other' in self.namespace:
            return all((x == self.target_namespace for x in namespaces))
        else:
            return all((x not in self.namespace for x in namespaces))

    def deny_qnames(self, names: Iterable[str]) -> bool:
        if self.not_namespace:
            return all((x in self.not_qname or get_namespace(x) in self.not_namespace for x in names))
        elif '##any' in self.namespace:
            return all((x in self.not_qname for x in names))
        elif '##other' in self.namespace:
            return all((x in self.not_qname or get_namespace(x) == self.target_namespace for x in names))
        else:
            return all((x in self.not_qname or get_namespace(x) not in self.namespace for x in names))

    def is_restriction(self, other: Union[ModelParticleType, 'XsdAnyAttribute'], check_occurs: bool=True) -> bool:
        if not isinstance(other, self.__class__):
            return False
        elif check_occurs and isinstance(self, ParticleMixin):
            if not isinstance(other, XsdAnyAttribute) and (not self.has_occurs_restriction(other)):
                return False
            elif self.max_occurs == 0:
                return True
        other: XsdWildcard
        if other.process_contents == 'strict' and self.process_contents != 'strict':
            return False
        elif other.process_contents == 'lax' and self.process_contents == 'skip':
            return False
        if not self.not_qname and (not other.not_qname):
            pass
        elif '##defined' in other.not_qname and '##defined' not in self.not_qname:
            return False
        elif '##definedSibling' in other.not_qname and '##definedSibling' not in self.not_qname:
            return False
        elif other.not_qname:
            if not self.deny_qnames((x for x in other.not_qname if not x.startswith('##'))):
                return False
        elif any((not other.is_namespace_allowed(get_namespace(x)) for x in self.not_qname if not x.startswith('##'))):
            return False
        if self.not_namespace:
            if other.not_namespace:
                return all((ns in self.not_namespace for ns in other.not_namespace))
            elif '##any' in other.namespace:
                return True
            elif '##other' in other.namespace:
                return '' in self.not_namespace and other.target_namespace in self.not_namespace
            else:
                return False
        elif other.not_namespace:
            if '##any' in self.namespace:
                return False
            elif '##other' in self.namespace:
                return set(other.not_namespace).issubset({'', other.target_namespace})
            else:
                return all((ns not in other.not_namespace for ns in self.namespace))
        if self.namespace == other.namespace:
            return True
        elif '##any' in other.namespace:
            return True
        elif '##any' in self.namespace or '##other' in self.namespace:
            return False
        elif '##other' in other.namespace:
            return other.target_namespace not in self.namespace and '' not in self.namespace
        else:
            return all((ns in other.namespace for ns in self.namespace))

    def union(self, other: Union['XsdAnyElement', 'XsdAnyAttribute']) -> None:
        """
        Update an XSD wildcard with the union of itself and another XSD wildcard.
        """
        if not self.not_qname:
            self.not_qname = other.not_qname[:]
        else:
            self.not_qname = [x for x in self.not_qname if x in other.not_qname or not other.is_namespace_allowed(get_namespace(x))]
        if self.not_namespace:
            if other.not_namespace:
                self.not_namespace = [ns for ns in self.not_namespace if ns in other.not_namespace]
            elif '##any' in other.namespace:
                self.not_namespace = []
                self.namespace = ['##any']
                return
            elif '##other' in other.namespace:
                not_namespace = ('', other.target_namespace)
                self.not_namespace = [ns for ns in self.not_namespace if ns in not_namespace]
            else:
                self.not_namespace = [ns for ns in self.not_namespace if ns not in other.namespace]
            if not self.not_namespace:
                self.namespace = ['##any']
            return
        elif other.not_namespace:
            if '##any' in self.namespace:
                return
            elif '##other' in self.namespace:
                not_namespace = ('', self.target_namespace)
                self.not_namespace = [ns for ns in other.not_namespace if ns in not_namespace]
            else:
                self.not_namespace = [ns for ns in other.not_namespace if ns not in self.namespace]
            self.namespace = ['##any'] if not self.not_namespace else []
            return
        w1: XsdWildcard
        w2: XsdWildcard
        if '##any' in self.namespace or self.namespace == other.namespace:
            return
        elif '##any' in other.namespace:
            self.namespace = ['##any']
            return
        elif '##other' in other.namespace:
            w1, w2 = (other, self)
        elif '##other' in self.namespace:
            w1, w2 = (self, other)
        else:
            assert isinstance(self.namespace, list)
            self.namespace.extend((ns for ns in other.namespace if ns not in self.namespace))
            return
        if w1.target_namespace in w2.namespace and '' in w2.namespace:
            self.namespace = ['##any']
        elif '' not in w2.namespace and w1.target_namespace == w2.target_namespace:
            self.namespace = ['##other']
        elif self.xsd_version == '1.0':
            msg = _('not expressible wildcard namespace union: {0!r} V {1!r}:')
            raise XMLSchemaValueError(msg.format(other.namespace, self.namespace))
        else:
            self.namespace = []
            self.not_namespace = ['', w1.target_namespace] if w1.target_namespace else ['']

    def intersection(self, other: Union['XsdAnyElement', 'XsdAnyAttribute']) -> None:
        """
        Update an XSD wildcard with the intersection of itself and another XSD wildcard.
        """
        if self.not_qname:
            self.not_qname.extend((x for x in other.not_qname if x not in self.not_qname))
        else:
            self.not_qname = [x for x in other.not_qname]
        if self.not_namespace:
            if other.not_namespace:
                self.not_namespace.extend((ns for ns in other.not_namespace if ns not in self.not_namespace))
            elif '##any' in other.namespace:
                pass
            elif '##other' not in other.namespace:
                self.namespace = [ns for ns in other.namespace if ns not in self.not_namespace]
                self.not_namespace = []
            else:
                if other.target_namespace not in self.not_namespace:
                    self.not_namespace.append(other.target_namespace)
                if '' not in self.not_namespace:
                    self.not_namespace.append('')
            return
        elif other.not_namespace:
            if '##any' in self.namespace:
                self.not_namespace = [ns for ns in other.not_namespace]
                self.namespace = []
            elif '##other' not in self.namespace:
                self.namespace = [ns for ns in self.namespace if ns not in other.not_namespace]
            else:
                self.not_namespace = [ns for ns in other.not_namespace]
                if self.target_namespace not in self.not_namespace:
                    self.not_namespace.append(self.target_namespace)
                if '' not in self.not_namespace:
                    self.not_namespace.append('')
                self.namespace = []
            return
        if self.namespace == other.namespace:
            return
        elif '##any' in other.namespace:
            return
        elif '##any' in self.namespace:
            self.namespace = other.namespace[:]
        elif '##other' in self.namespace:
            self.namespace = [ns for ns in other.namespace if ns not in ('', self.target_namespace)]
        elif '##other' not in other.namespace:
            self.namespace = [ns for ns in self.namespace if ns in other.namespace]
        else:
            assert isinstance(self.namespace, list)
            if other.target_namespace in self.namespace:
                self.namespace.remove(other.target_namespace)
            if '' in self.namespace:
                self.namespace.remove('')