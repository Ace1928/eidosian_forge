from abc import ABCMeta
import os
import logging
import threading
import warnings
import re
import sys
from copy import copy as _copy
from operator import attrgetter
from typing import cast, Callable, ItemsView, List, Optional, Dict, Any, \
from xml.etree.ElementTree import Element, ParseError
from elementpath import XPathToken, SchemaElementNode, build_schema_node_tree
from ..exceptions import XMLSchemaTypeError, XMLSchemaKeyError, XMLSchemaRuntimeError, \
from ..names import VC_MIN_VERSION, VC_MAX_VERSION, VC_TYPE_AVAILABLE, \
from ..aliases import ElementType, XMLSourceType, NamespacesType, LocationsType, \
from ..translation import gettext as _
from ..helpers import prune_etree, get_namespace, get_qname, is_defuse_error
from ..namespaces import NamespaceResourcesMap, NamespaceView
from ..resources import is_local_url, is_remote_url, url_path_is_file, \
from ..converters import XMLSchemaConverter
from ..xpath import XsdSchemaProtocol, XMLSchemaProxy, ElementPathMixin
from .. import dataobjects
from .exceptions import XMLSchemaParseError, XMLSchemaValidationError, XMLSchemaEncodeError, \
from .helpers import get_xsd_derivation_attribute
from .xsdbase import check_validation_mode, XsdValidator, XsdComponent, XsdAnnotation
from .notations import XsdNotation
from .identities import XsdIdentity, XsdKey, XsdKeyref, XsdUnique, \
from .facets import XSD_10_FACETS, XSD_11_FACETS
from .simple_types import XsdSimpleType, XsdList, XsdUnion, XsdAtomicRestriction, \
from .attributes import XsdAttribute, XsdAttributeGroup, Xsd11Attribute
from .complex_types import XsdComplexType, Xsd11ComplexType
from .groups import XsdGroup, Xsd11Group
from .elements import XsdElement, Xsd11Element
from .wildcards import XsdAnyElement, XsdAnyAttribute, Xsd11AnyElement, \
from .global_maps import XsdGlobals
def add_schema(self, source: SchemaSourceType, namespace: Optional[str]=None, build: bool=False) -> SchemaType:
    """
        Add another schema source to the maps of the instance.

        :param source: a URI that reference to a resource or a file path or a file-like         object or a string containing the schema or an Element or an ElementTree document.
        :param namespace: is an optional argument that contains the URI of the namespace         that has to used in case the schema has no namespace (chameleon schema). For other         cases, when specified, it must be equal to the *targetNamespace* of the schema.
        :param build: defines when to build the imported schema, the default is to not build.
        :return: the added :class:`XMLSchema` instance.
        """
    return type(self)(source=source, namespace=namespace, validation=self.validation, global_maps=self.maps, converter=self.converter, locations=[x for x in self._locations if x[0] != namespace], base_url=self.base_url, allow=self.allow, defuse=self.defuse, timeout=self.timeout, build=build)