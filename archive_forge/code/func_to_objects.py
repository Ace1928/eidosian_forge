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
def to_objects(self, source: Union[XMLSourceType, XMLResource], with_bindings: bool=False, **kwargs: Any) -> DecodeType['dataobjects.DataElement']:
    """
        Decodes XML data to Python data objects.

        :param source: the XML data. Can be a string for an attribute or for a simple         type components or a dictionary for an attribute group or an ElementTree's         Element for other components.
        :param with_bindings: if `True` is provided the decoding is done using         :class:`DataBindingConverter` that used XML data binding classes. For         default the objects are instances of :class:`DataElement` and uses the         :class:`DataElementConverter`.
        :param kwargs: other optional keyword arguments for the method         :func:`iter_decode`, except the argument *converter*.
        """
    if with_bindings:
        return self.decode(source, converter=dataobjects.DataBindingConverter, **kwargs)
    return self.decode(source, converter=dataobjects.DataElementConverter, **kwargs)