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
def _import_namespace(self, namespace: str, locations: List[str]) -> None:
    import_error: Optional[Exception] = None
    for url in locations:
        try:
            logger.debug('Import namespace %r from %r', namespace, url)
            self.import_schema(namespace, url, self.base_url)
        except (OSError, IOError) as err:
            logger.debug('%s', err)
            if import_error is None:
                import_error = err
        except (XMLSchemaParseError, XMLSchemaTypeError, ParseError) as err:
            if is_defuse_error(err):
                logger.debug('%s', err)
                if import_error is None:
                    import_error = err
            else:
                if namespace:
                    msg = _('cannot import namespace {0!r}: {1}').format(namespace, err)
                else:
                    msg = _('cannot import chameleon schema: %s') % err
                if isinstance(err, (XMLSchemaParseError, ParseError)):
                    self.parse_error(msg)
                else:
                    raise type(err)(msg)
        except XMLSchemaValueError as err:
            self.parse_error(err)
        else:
            logger.info('Namespace %r imported from %r', namespace, url)
            break
    else:
        if import_error is not None:
            msg = 'Import of namespace {!r} from {!r} failed: {}.'
            self.warnings.append(msg.format(namespace, locations, str(import_error)))
            warnings.warn(self.warnings[-1], XMLSchemaImportWarning, stacklevel=4)
        self.imports[namespace] = None