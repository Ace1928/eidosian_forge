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
class XsdAnyElement(XsdWildcard, ParticleMixin, ElementPathMixin[SchemaElementType], ValidationMixin[ElementType, Any]):
    """
    Class for XSD 1.0 *any* wildcards.

    ..  <any
          id = ID
          maxOccurs = (nonNegativeInteger | unbounded) : 1
          minOccurs = nonNegativeInteger : 1
          namespace = ((##any | ##other) | List of (anyURI | (##targetNamespace|##local)) ) : ##any
          processContents = (lax | skip | strict) : strict
          {any attributes with non-schema namespace . . .}>
          Content: (annotation?)
        </any>
    """
    _ADMITTED_TAGS = {XSD_ANY}
    precedences: Dict[ModelGroupType, List[ModelParticleType]]
    copy: Callable[['XsdAnyElement'], 'XsdAnyElement']

    def __init__(self, elem: ElementType, schema: SchemaType, parent: XsdComponent) -> None:
        self.precedences = {}
        super(XsdAnyElement, self).__init__(elem, schema, parent)

    def __repr__(self) -> str:
        if self.namespace:
            return '%s(namespace=%r, process_contents=%r, occurs=%r)' % (self.__class__.__name__, self.namespace, self.process_contents, list(self.occurs))
        else:
            return '%s(not_namespace=%r, process_contents=%r, occurs=%r)' % (self.__class__.__name__, self.not_namespace, self.process_contents, list(self.occurs))

    @property
    def xpath_proxy(self) -> XMLSchemaProxy:
        return XMLSchemaProxy(schema=cast(XsdSchemaProtocol, self.schema), base_element=cast(XsdElementProtocol, self))

    @property
    def xpath_node(self) -> SchemaElementNode:
        schema_node = self.schema.xpath_node
        node = schema_node.get_element_node(cast(XsdElementProtocol, self))
        if isinstance(node, SchemaElementNode):
            return node
        return build_schema_node_tree(root=cast(XsdElementProtocol, self), elements=schema_node.elements, global_elements=schema_node.children)

    def _parse(self) -> None:
        super(XsdAnyElement, self)._parse()
        self._parse_particle(self.elem)

    def match(self, name: Optional[str], default_namespace: Optional[str]=None, resolve: bool=False, **kwargs: Any) -> Optional[SchemaElementType]:
        """
        Returns the element wildcard if name is matching the name provided
        as argument, `None` otherwise.

        :param name: a local or fully-qualified name.
        :param default_namespace: used when it's not `None` and not empty for         completing local name arguments.
        :param resolve: when `True` it doesn't return the wildcard but try to         resolve and return the element matching the name.
        :param kwargs: additional options used by XSD 1.1 xs:any wildcards.
        """
        if not name or not self.is_matching(name, default_namespace, **kwargs):
            return None
        elif not resolve:
            return self
        try:
            if name[0] != '{' and default_namespace:
                return self.maps.lookup_element(f'{{{default_namespace}}}{name}')
            else:
                return self.maps.lookup_element(name)
        except LookupError:
            return None

    def __iter__(self) -> Iterator[Any]:
        return iter(())

    def iter(self, tag: Optional[str]=None) -> Iterator[Any]:
        return iter(())

    def iterchildren(self, tag: Optional[str]=None) -> Iterator[Any]:
        return iter(())

    @staticmethod
    def iter_substitutes() -> Iterator[Any]:
        return iter(())

    def iter_decode(self, obj: ElementType, validation: str='lax', **kwargs: Any) -> IterDecodeType[Any]:
        if not self.is_matching(obj.tag):
            reason = _('element {!r} is not allowed here').format(obj)
            yield self.validation_error(validation, reason, obj, **kwargs)
        if self.process_contents == 'skip':
            if 'process_skipped' not in kwargs or not kwargs['process_skipped']:
                return
        namespace = get_namespace(obj.tag)
        if not self.maps.load_namespace(namespace):
            reason = f'unavailable namespace {namespace!r}'
        else:
            try:
                xsd_element = self.maps.lookup_element(obj.tag)
            except LookupError:
                reason = f'element {obj.tag!r} not found'
            else:
                yield from xsd_element.iter_decode(obj, validation, **kwargs)
                return
        if XSI_TYPE in obj.attrib:
            if self.process_contents == 'strict':
                xsd_element = self.maps.validator.create_element(obj.tag, parent=self, form='unqualified')
            else:
                xsd_element = self.maps.validator.create_element(obj.tag, parent=self, nillable='true', form='unqualified')
            yield from xsd_element.iter_decode(obj, validation, **kwargs)
            return
        if validation != 'skip' and self.process_contents == 'strict':
            yield self.validation_error(validation, reason, obj, **kwargs)
        yield from self.any_type.iter_decode(obj, validation, **kwargs)

    def iter_encode(self, obj: Tuple[str, ElementType], validation: str='lax', **kwargs: Any) -> IterEncodeType[Any]:
        name, value = obj
        namespace = get_namespace(name)
        if not self.is_namespace_allowed(namespace):
            reason = _('element {!r} is not allowed here').format(name)
            yield self.validation_error(validation, reason, value, **kwargs)
        if self.process_contents == 'skip':
            if 'process_skipped' not in kwargs or not kwargs['process_skipped']:
                return
        if not self.maps.load_namespace(namespace):
            reason = f'unavailable namespace {namespace!r}'
        else:
            try:
                xsd_element = self.maps.lookup_element(name)
            except LookupError:
                reason = f'element {name!r} not found'
            else:
                yield from xsd_element.iter_encode(value, validation, **kwargs)
                return
        if self.process_contents == 'strict':
            xsd_element = self.maps.validator.create_element(name, parent=self, form='unqualified')
        else:
            xsd_element = self.maps.validator.create_element(name, parent=self, nillable='true', form='unqualified')
        try:
            converter = kwargs['converter']
        except KeyError:
            converter = kwargs['converter'] = self.schema.get_converter(**kwargs)
        try:
            level = kwargs['level']
        except KeyError:
            level = 0
        try:
            element_data = converter.element_encode(value, xsd_element, level)
        except (ValueError, TypeError) as err:
            if validation != 'skip' and self.process_contents == 'strict':
                yield self.validation_error(validation, err, value, **kwargs)
        else:
            if XSI_TYPE in element_data.attributes:
                yield from xsd_element.iter_encode(value, validation, **kwargs)
                return
        if validation != 'skip' and self.process_contents == 'strict':
            yield self.validation_error(validation, reason, **kwargs)
        yield from self.any_type.iter_encode(obj, validation, **kwargs)

    def is_overlap(self, other: ModelParticleType) -> bool:
        if not isinstance(other, XsdAnyElement):
            if isinstance(other, elements.XsdElement):
                return other.is_overlap(self)
            return False
        if self.not_namespace:
            if other.not_namespace:
                return True
            elif '##any' in other.namespace:
                return True
            elif '##other' in other.namespace:
                return True
            else:
                return any((ns not in self.not_namespace for ns in other.namespace))
        elif other.not_namespace:
            if '##any' in self.namespace:
                return True
            elif '##other' in self.namespace:
                return True
            else:
                return any((ns not in other.not_namespace for ns in self.namespace))
        elif self.namespace == other.namespace:
            return True
        elif '##any' in self.namespace or '##any' in other.namespace:
            return True
        elif '##other' in self.namespace:
            return any((ns and ns != self.target_namespace for ns in other.namespace))
        elif '##other' in other.namespace:
            return any((ns and ns != other.target_namespace for ns in self.namespace))
        else:
            return any((ns in self.namespace for ns in other.namespace))

    def is_consistent(self, other: SchemaElementType, **kwargs: Any) -> bool:
        return True