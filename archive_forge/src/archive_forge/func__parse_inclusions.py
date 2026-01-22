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
def _parse_inclusions(self) -> None:
    """Processes schema document inclusions and redefinitions/overrides."""
    for child in self.source.root:
        if 'schemaLocation' not in child.attrib:
            continue
        location = child.attrib['schemaLocation'].strip()
        if child.tag == XSD_INCLUDE:
            try:
                logger.info('Include schema from %r', location)
                self.include_schema(location, self.base_url)
            except (OSError, IOError) as err:
                self.warnings.append('Include schema failed: %s.' % str(err))
                warnings.warn(self.warnings[-1], XMLSchemaIncludeWarning, stacklevel=3)
            except (XMLSchemaParseError, XMLSchemaTypeError, ParseError) as err:
                msg = _('cannot include schema {0!r}: {1}')
                if isinstance(err, (XMLSchemaParseError, ParseError)):
                    self.parse_error(msg.format(location, err), child)
                else:
                    raise type(err)(msg.format(location, err))
        elif child.tag == XSD_REDEFINE:
            try:
                logger.info('Redefine schema %r', location)
                schema = self.include_schema(location, self.base_url)
            except (OSError, IOError) as err:
                self.warnings.append(_('Redefine schema failed: %s') % str(err))
                warnings.warn(self.warnings[-1], XMLSchemaIncludeWarning, stacklevel=3)
                if any((e.tag != XSD_ANNOTATION and (not callable(e.tag)) for e in child)):
                    self.parse_error(err, child)
            except (XMLSchemaParseError, XMLSchemaTypeError, ParseError) as err:
                msg = _('cannot redefine schema {0!r}: {1}')
                if isinstance(err, (XMLSchemaParseError, ParseError)):
                    self.parse_error(msg.format(location, err), child)
                else:
                    raise type(err)(msg.format(location, err))
            else:
                schema.redefine = self
        elif child.tag == XSD_OVERRIDE and self.XSD_VERSION != '1.0':
            try:
                logger.info('Override schema %r', location)
                schema = self.include_schema(location, self.base_url)
            except (OSError, IOError) as err:
                self.warnings.append(_('Override schema failed: %s') % str(err))
                warnings.warn(self.warnings[-1], XMLSchemaIncludeWarning, stacklevel=3)
                if any((e.tag != XSD_ANNOTATION and (not callable(e.tag)) for e in child)):
                    self.parse_error(str(err), child)
            else:
                schema.override = self