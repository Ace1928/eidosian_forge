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
def version_check(self, elem: ElementType) -> bool:
    """
        Checks if the element is compatible with the version of the validator and XSD
        types/facets availability. Invalid vc attributes are not detected in XSD 1.0.

        :param elem: an Element of the schema.
        :return: `True` if the schema element is compatible with the validator,         `False` otherwise.
        """
    if VC_MIN_VERSION in elem.attrib:
        vc_min_version = elem.attrib[VC_MIN_VERSION]
        if not XSD_VERSION_PATTERN.match(vc_min_version):
            if self.XSD_VERSION > '1.0':
                msg = _('invalid attribute vc:minVersion value')
                self.parse_error(msg, elem)
        elif vc_min_version > self.XSD_VERSION:
            return False
    if VC_MAX_VERSION in elem.attrib:
        vc_max_version = elem.attrib[VC_MAX_VERSION]
        if not XSD_VERSION_PATTERN.match(vc_max_version):
            if self.XSD_VERSION > '1.0':
                msg = _('invalid attribute vc:maxVersion value')
                self.parse_error(msg, elem)
        elif vc_max_version <= self.XSD_VERSION:
            return False
    if VC_TYPE_AVAILABLE in elem.attrib:
        for qname in elem.attrib[VC_TYPE_AVAILABLE].split():
            try:
                if self.resolve_qname(qname) not in self.maps.types:
                    return False
            except XMLSchemaNamespaceError:
                return False
            except (KeyError, ValueError) as err:
                self.parse_error(str(err), elem)
    if VC_TYPE_UNAVAILABLE in elem.attrib:
        for qname in elem.attrib[VC_TYPE_UNAVAILABLE].split():
            try:
                if self.resolve_qname(qname) not in self.maps.types:
                    break
            except XMLSchemaNamespaceError:
                break
            except (KeyError, ValueError) as err:
                self.parse_error(err, elem)
        else:
            return False
    if VC_FACET_AVAILABLE in elem.attrib:
        for qname in elem.attrib[VC_FACET_AVAILABLE].split():
            try:
                facet_name = self.resolve_qname(qname)
            except XMLSchemaNamespaceError:
                pass
            except (KeyError, ValueError) as err:
                self.parse_error(str(err), elem)
            else:
                if self.XSD_VERSION == '1.0':
                    if facet_name not in XSD_10_FACETS:
                        return False
                elif facet_name not in XSD_11_FACETS:
                    return False
    if VC_FACET_UNAVAILABLE in elem.attrib:
        for qname in elem.attrib[VC_FACET_UNAVAILABLE].split():
            try:
                facet_name = self.resolve_qname(qname)
            except XMLSchemaNamespaceError:
                break
            except (KeyError, ValueError) as err:
                self.parse_error(err, elem)
            else:
                if self.XSD_VERSION == '1.0':
                    if facet_name not in XSD_10_FACETS:
                        break
                elif facet_name not in XSD_11_FACETS:
                    break
        else:
            return False
    return True