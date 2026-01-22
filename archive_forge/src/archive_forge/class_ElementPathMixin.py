import sys
from abc import abstractmethod
from typing import cast, overload, Any, Dict, Iterator, List, Optional, \
import re
from elementpath import XPath2Parser, XPathSchemaContext, \
from .exceptions import XMLSchemaValueError, XMLSchemaTypeError
from .names import XSD_NAMESPACE
from .aliases import NamespacesType, SchemaType, BaseXsdType, XPathElementType
from .helpers import get_qname, local_name, get_prefixed_qname
class ElementPathMixin(Sequence[E]):
    """
    Mixin abstract class for enabling ElementTree and XPath 2.0 API on XSD components.

    :cvar text: the Element text, for compatibility with the ElementTree API.
    :cvar tail: the Element tail, for compatibility with the ElementTree API.
    """
    text: Optional[str] = None
    tail: Optional[str] = None
    name: Optional[str] = None
    attributes: Any = {}
    namespaces: Any = {}
    xpath_default_namespace = ''
    _xpath_node: Optional[Union[SchemaElementNode, LazyElementNode]] = None

    @abstractmethod
    def __iter__(self) -> Iterator[E]:
        raise NotImplementedError

    @overload
    def __getitem__(self, i: int) -> E:
        ...

    @overload
    def __getitem__(self, s: slice) -> Sequence[E]:
        ...

    def __getitem__(self, i: Union[int, slice]) -> Union[E, Sequence[E]]:
        try:
            return [e for e in self][i]
        except IndexError:
            raise IndexError('child index out of range')

    def __reversed__(self) -> Iterator[E]:
        return reversed([e for e in self])

    def __len__(self) -> int:
        return len([e for e in self])

    @property
    def tag(self) -> str:
        """Alias of the *name* attribute. For compatibility with the ElementTree API."""
        return self.name or ''

    @property
    def attrib(self) -> Any:
        """Returns the Element attributes. For compatibility with the ElementTree API."""
        return self.attributes

    def get(self, key: str, default: Any=None) -> Any:
        """Gets an Element attribute. For compatibility with the ElementTree API."""
        return self.attributes.get(key, default)

    @property
    def xpath_proxy(self) -> XMLSchemaProxy:
        """Returns an XPath proxy instance bound with the schema."""
        raise NotImplementedError

    @property
    def xpath_node(self) -> Union[SchemaElementNode, LazyElementNode]:
        """Returns an XPath node for applying selectors on XSD schema/component."""
        raise NotImplementedError

    def _get_xpath_namespaces(self, namespaces: Optional[NamespacesType]=None) -> Dict[str, str]:
        """
        Returns a dictionary with namespaces for XPath selection.

        :param namespaces: an optional map from namespace prefix to namespace URI.         If this argument is not provided the schema's namespaces are used.
        """
        xpath_namespaces: Dict[str, str] = XPath2Parser.DEFAULT_NAMESPACES.copy()
        if namespaces is None:
            xpath_namespaces.update(self.namespaces)
        else:
            xpath_namespaces.update(namespaces)
        return xpath_namespaces

    def is_matching(self, name: Optional[str], default_namespace: Optional[str]=None) -> bool:
        if not name or name[0] == '{' or (not default_namespace):
            return self.name == name
        else:
            return self.name == f'{{{default_namespace}}}{name}'

    def find(self, path: str, namespaces: Optional[NamespacesType]=None) -> Optional[E]:
        """
        Finds the first XSD subelement matching the path.

        :param path: an XPath expression that considers the XSD component as the root element.
        :param namespaces: an optional mapping from namespace prefix to namespace URI.
        :return: the first matching XSD subelement or ``None`` if there is no match.
        """
        path = _REGEX_TAG_POSITION.sub('', path.strip())
        namespaces = self._get_xpath_namespaces(namespaces)
        parser = XPath2Parser(namespaces, strict=False)
        context = XPathSchemaContext(self.xpath_node)
        return cast(Optional[E], next(parser.parse(path).select_results(context), None))

    def findall(self, path: str, namespaces: Optional[NamespacesType]=None) -> List[E]:
        """
        Finds all XSD subelements matching the path.

        :param path: an XPath expression that considers the XSD component as the root element.
        :param namespaces: an optional mapping from namespace prefix to full name.
        :return: a list containing all matching XSD subelements in document order, an empty         list is returned if there is no match.
        """
        path = _REGEX_TAG_POSITION.sub('', path.strip())
        namespaces = self._get_xpath_namespaces(namespaces)
        parser = XPath2Parser(namespaces, strict=False)
        context = XPathSchemaContext(self.xpath_node)
        return cast(List[E], parser.parse(path).get_results(context))

    def iterfind(self, path: str, namespaces: Optional[NamespacesType]=None) -> Iterator[E]:
        """
        Creates and iterator for all XSD subelements matching the path.

        :param path: an XPath expression that considers the XSD component as the root element.
        :param namespaces: is an optional mapping from namespace prefix to full name.
        :return: an iterable yielding all matching XSD subelements in document order.
        """
        path = _REGEX_TAG_POSITION.sub('', path.strip())
        namespaces = self._get_xpath_namespaces(namespaces)
        parser = XPath2Parser(namespaces, strict=False)
        context = XPathSchemaContext(self.xpath_node)
        return cast(Iterator[E], parser.parse(path).select_results(context))

    def iter(self, tag: Optional[str]=None) -> Iterator[E]:
        """
        Creates an iterator for the XSD element and its subelements. If tag is not `None` or '*',
        only XSD elements whose matches tag are returned from the iterator. Local elements are
        expanded without repetitions. Element references are not expanded because the global
        elements are not descendants of other elements.
        """

        def safe_iter(elem: Any) -> Iterator[E]:
            if tag is None or elem.is_matching(tag):
                yield elem
            for child in elem:
                if child.parent is None:
                    yield from safe_iter(child)
                elif getattr(child, 'ref', None) is not None:
                    if tag is None or child.is_matching(tag):
                        yield child
                elif child not in local_elements:
                    local_elements.add(child)
                    yield from safe_iter(child)
        if tag == '*':
            tag = None
        local_elements: Set[E] = set()
        return safe_iter(self)

    def iterchildren(self, tag: Optional[str]=None) -> Iterator[E]:
        """
        Creates an iterator for the child elements of the XSD component. If *tag* is not `None`
        or '*', only XSD elements whose name matches tag are returned from the iterator.
        """
        if tag == '*':
            tag = None
        for child in self:
            if tag is None or child.is_matching(tag):
                yield child