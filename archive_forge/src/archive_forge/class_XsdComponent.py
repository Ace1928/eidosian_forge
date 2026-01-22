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
class XsdComponent(XsdValidator):
    """
    Class for XSD components. See: https://www.w3.org/TR/xmlschema-ref/

    :param elem: ElementTree's node containing the definition.
    :param schema: the XMLSchema object that owns the definition.
    :param parent: the XSD parent, `None` means that is a global component that     has the schema as parent.
    :param name: name of the component, maybe overwritten by the parse of the `elem` argument.

    :cvar qualified: for name matching, unqualified matching may be admitted only     for elements and attributes.
    :vartype qualified: bool
    """
    _REGEX_SPACE = re.compile('\\s')
    _REGEX_SPACES = re.compile('\\s+')
    _ADMITTED_TAGS: Union[Set[str], Tuple[str, ...], Tuple[()]] = ()
    elem: ElementTree.Element
    parent = None
    name = None
    ref: Optional['XsdComponent'] = None
    qualified = True
    redefine = None
    _annotation = None
    _target_namespace: Optional[str]

    def __init__(self, elem: ElementTree.Element, schema: SchemaType, parent: Optional['XsdComponent']=None, name: Optional[str]=None) -> None:
        super(XsdComponent, self).__init__(schema.validation)
        if name:
            self.name = name
        if parent is not None:
            self.parent = parent
        self.schema = schema
        self.maps: XsdGlobals = schema.maps
        self.elem = elem

    def __setattr__(self, name: str, value: Any) -> None:
        if name == 'elem':
            if value.tag not in self._ADMITTED_TAGS:
                msg = 'wrong XSD element {!r} for {!r}, must be one of {!r}'
                raise XMLSchemaValueError(msg.format(value.tag, self.__class__, self._ADMITTED_TAGS))
            super(XsdComponent, self).__setattr__(name, value)
            if self.errors:
                self.errors.clear()
            self._parse()
        else:
            super(XsdComponent, self).__setattr__(name, value)

    @property
    def xsd_version(self) -> str:
        return self.schema.XSD_VERSION

    def is_global(self) -> bool:
        """Returns `True` if the instance is a global component, `False` if it's local."""
        return self.parent is None

    def is_override(self) -> bool:
        """Returns `True` if the instance is an override of a global component."""
        if self.parent is not None:
            return False
        return any((self.elem in x for x in self.schema.root if x.tag == XSD_OVERRIDE))

    @property
    def schema_elem(self) -> ElementType:
        """The reference element of the schema for the component instance."""
        return self.elem

    @property
    def source(self) -> XMLResource:
        """Property that references to schema source."""
        return self.schema.source

    @property
    def target_namespace(self) -> str:
        """Property that references to schema's targetNamespace."""
        return self.schema.target_namespace if self.ref is None else self.ref.target_namespace

    @property
    def default_namespace(self) -> Optional[str]:
        """Property that references to schema's default namespaces."""
        return self.schema.namespaces.get('')

    @property
    def namespaces(self) -> NamespacesType:
        """Property that references to schema's namespace mapping."""
        return self.schema.namespaces

    @property
    def any_type(self) -> 'XsdComplexType':
        """Property that references to the xs:anyType instance of the global maps."""
        return cast('XsdComplexType', self.maps.types[XSD_ANY_TYPE])

    @property
    def any_simple_type(self) -> 'XsdSimpleType':
        """Property that references to the xs:anySimpleType instance of the global maps."""
        return cast('XsdSimpleType', self.maps.types[XSD_ANY_SIMPLE_TYPE])

    @property
    def any_atomic_type(self) -> 'XsdSimpleType':
        """Property that references to the xs:anyAtomicType instance of the global maps."""
        return cast('XsdSimpleType', self.maps.types[XSD_ANY_ATOMIC_TYPE])

    @property
    def annotation(self) -> Optional['XsdAnnotation']:
        if '_annotation' not in self.__dict__:
            for child in self.elem:
                if child.tag == XSD_ANNOTATION:
                    self._annotation = XsdAnnotation(child, self.schema, self)
                    break
                elif not callable(child.tag):
                    self._annotation = None
                    break
            else:
                self._annotation = None
        return self._annotation

    def __repr__(self) -> str:
        if self.ref is not None:
            return '%s(ref=%r)' % (self.__class__.__name__, self.prefixed_name)
        else:
            return '%s(name=%r)' % (self.__class__.__name__, self.prefixed_name)

    def _parse(self) -> None:
        return

    def _parse_reference(self) -> Optional[bool]:
        """
        Helper method for referable components. Returns `True` if a valid reference QName
        is found without any error, otherwise returns `None`. Sets an id-related name for
        the component ('nameless_<id of the instance>') if both the attributes 'ref' and
        'name' are missing.
        """
        ref = self.elem.get('ref')
        if ref is None:
            if 'name' in self.elem.attrib:
                return None
            elif self.parent is None:
                msg = _("missing attribute 'name' in a global %r")
                self.parse_error(msg % type(self))
            else:
                msg = _("missing both attributes 'name' and 'ref' in local %r")
                self.parse_error(msg % type(self))
        elif 'name' in self.elem.attrib:
            msg = _("attributes 'name' and 'ref' are mutually exclusive")
            self.parse_error(msg)
        elif self.parent is None:
            msg = _("attribute 'ref' not allowed in a global %r")
            self.parse_error(msg % type(self))
        else:
            try:
                self.name = self.schema.resolve_qname(ref)
            except (KeyError, ValueError, RuntimeError) as err:
                self.parse_error(err)
            else:
                if self._parse_child_component(self.elem, strict=False) is not None:
                    msg = _('a reference component cannot have child definitions/declarations')
                    self.parse_error(msg)
                return True
        return None

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

    def _parse_target_namespace(self) -> None:
        """
        XSD 1.1 targetNamespace attribute in elements and attributes declarations.
        """
        if 'targetNamespace' not in self.elem.attrib:
            return
        self._target_namespace = self.elem.attrib['targetNamespace'].strip()
        if 'name' not in self.elem.attrib:
            msg = _("attribute 'name' must be present when 'targetNamespace' attribute is provided")
            self.parse_error(msg)
        if 'form' in self.elem.attrib:
            msg = _("attribute 'form' must be absent when 'targetNamespace' attribute is provided")
            self.parse_error(msg)
        if self._target_namespace != self.schema.target_namespace:
            if self.parent is None:
                msg = _('a global %s must have the same namespace as its parent schema')
                self.parse_error(msg % self.__class__.__name__)
            xsd_type = self.get_parent_type()
            if xsd_type is None or xsd_type.parent is not None:
                pass
            elif xsd_type.derivation != 'restriction' or getattr(xsd_type.base_type, 'name', None) == XSD_ANY_TYPE:
                msg = _('a declaration contained in a global complexType must have the same namespace as its parent schema')
                self.parse_error(msg)
        if self.name is None:
            pass
        elif not self._target_namespace:
            self.name = local_name(self.name)
        else:
            self.name = f'{{{self._target_namespace}}}{local_name(self.name)}'

    @property
    def local_name(self) -> Optional[str]:
        """The local part of the name of the component, or `None` if the name is `None`."""
        return None if self.name is None else local_name(self.name)

    @property
    def qualified_name(self) -> Optional[str]:
        """The name of the component in extended format, or `None` if the name is `None`."""
        return None if self.name is None else get_qname(self.target_namespace, self.name)

    @property
    def prefixed_name(self) -> Optional[str]:
        """The name of the component in prefixed format, or `None` if the name is `None`."""
        return None if self.name is None else get_prefixed_qname(self.name, self.namespaces)

    @property
    def id(self) -> Optional[str]:
        """The ``'id'`` attribute of the component tag, ``None`` if missing."""
        return self.elem.get('id')

    @property
    def validation_attempted(self) -> str:
        return 'full' if self.built else 'partial'

    def build(self) -> None:
        """
        Builds components that are not fully parsed at initialization, like model groups
        or internal local elements in model groups, otherwise does nothing.
        """

    @property
    def built(self) -> bool:
        raise NotImplementedError()

    def is_matching(self, name: Optional[str], default_namespace: Optional[str]=None, **kwargs: Any) -> bool:
        """
        Returns `True` if the component name is matching the name provided as argument,
        `False` otherwise. For XSD elements the matching is extended to substitutes.

        :param name: a local or fully-qualified name.
        :param default_namespace: used by the XPath processor for completing         the name argument in case it's a local name.
        :param kwargs: additional options that can be used by certain components.
        """
        return bool(self.name == name or (default_namespace and name and (name[0] != '{') and (self.name == f'{{{default_namespace}}}{name}')))

    def match(self, name: Optional[str], default_namespace: Optional[str]=None, **kwargs: Any) -> Optional['XsdComponent']:
        """
        Returns the component if its name is matching the name provided as argument,
        `None` otherwise.
        """
        return self if self.is_matching(name, default_namespace, **kwargs) else None

    def get_matching_item(self, mapping: MutableMapping[str, Any], ns_prefix: str='xmlns', match_local_name: bool=False) -> Optional[Any]:
        """
        If a key is matching component name, returns its value, otherwise returns `None`.
        """
        if self.name is None:
            return None
        elif not self.target_namespace:
            return mapping.get(self.name)
        elif self.qualified_name in mapping:
            return mapping[cast(str, self.qualified_name)]
        elif self.prefixed_name in mapping:
            return mapping[cast(str, self.prefixed_name)]
        target_namespace = self.target_namespace
        suffix = f':{self.local_name}'
        for k in filter(lambda x: x.endswith(suffix), mapping):
            prefix = k.split(':')[0]
            if self.namespaces.get(prefix) == target_namespace:
                return mapping[k]
            ns_declaration = '{}:{}'.format(ns_prefix, prefix)
            try:
                if mapping[k][ns_declaration] == target_namespace:
                    return mapping[k]
            except (KeyError, TypeError):
                pass
        else:
            if match_local_name:
                return mapping.get(self.local_name)
            return None

    def get_global(self) -> 'XsdComponent':
        """Returns the global XSD component that contains the component instance."""
        if self.parent is None:
            return self
        component = self.parent
        while component is not self:
            if component.parent is None:
                return component
            component = component.parent
        else:
            msg = _('parent circularity from {}')
            raise XMLSchemaValueError(msg.format(self))

    def get_parent_type(self) -> Optional['XsdType']:
        """
        Returns the nearest XSD type that contains the component instance,
        or `None` if the component doesn't have an XSD type parent.
        """
        component = self.parent
        while component is not self and component is not None:
            if isinstance(component, XsdType):
                return component
            component = component.parent
        return None

    def iter_components(self, xsd_classes: ComponentClassType=None) -> Iterator['XsdComponent']:
        """
        Creates an iterator for XSD subcomponents.

        :param xsd_classes: provide a class or a tuple of classes to iterate         over only a specific classes of components.
        """
        if xsd_classes is None or isinstance(self, xsd_classes):
            yield self

    def iter_ancestors(self, xsd_classes: ComponentClassType=None) -> Iterator['XsdComponent']:
        """
        Creates an iterator for XSD ancestor components, schema excluded.
        Stops when the component is global or if the ancestor is not an
        instance of the specified class/classes.

        :param xsd_classes: provide a class or a tuple of classes to iterate         over only a specific classes of components.
        """
        ancestor = self
        while True:
            if ancestor.parent is None:
                break
            ancestor = ancestor.parent
            if xsd_classes is not None and (not isinstance(ancestor, xsd_classes)):
                break
            yield ancestor

    def tostring(self, indent: str='', max_lines: Optional[int]=None, spaces_for_tab: int=4) -> Union[str, bytes]:
        """Serializes the XML elements that declare or define the component to a string."""
        return etree_tostring(self.schema_elem, self.namespaces, indent, max_lines, spaces_for_tab)