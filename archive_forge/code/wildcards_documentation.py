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

        Returns `True` if the component name is matching the name provided as argument,
        `False` otherwise.

        :param name: a local or fully-qualified name.
        :param default_namespace: used by the XPath processor for completing         the name argument in case it's a local name.
        :param group: used only by XSD 1.1 any element wildcards to verify siblings in         case of ##definedSibling value in notQName attribute.
        :param occurs: a Counter instance for verify model occurrences counting.
        