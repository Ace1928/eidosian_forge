import warnings
from collections import Counter
from functools import lru_cache
from typing import cast, Any, Callable, Dict, List, Iterable, Iterator, \
from ..exceptions import XMLSchemaKeyError, XMLSchemaTypeError, \
from ..names import XSD_NAMESPACE, XSD_REDEFINE, XSD_OVERRIDE, XSD_NOTATION, \
from ..aliases import ComponentClassType, ElementType, SchemaType, BaseXsdType, \
from ..helpers import get_qname, local_name, get_extended_qname
from ..namespaces import NamespaceResourcesMap
from ..translation import gettext as _
from .exceptions import XMLSchemaNotBuiltError, XMLSchemaModelError, XMLSchemaModelDepthError, \
from .xsdbase import XsdValidator, XsdComponent
from .builtins import xsd_builtin_types_factory
from .models import check_model
from . import XsdAttribute, XsdSimpleType, XsdComplexType, XsdElement, XsdAttributeGroup, \
def _load_namespace(self, namespace: str, build: bool=True) -> bool:
    """
        Load namespace from available location hints. Returns `True` if the namespace
        is already loaded or if the namespace can be loaded from one of the locations,
        returns `False` otherwise. Failing locations are inserted into the missing
        locations list.

        :param namespace: the namespace to load.
        :param build: if left with `True` value builds the maps after load. If the         build fails the resource URL is added to missing locations.
        """
    namespace = namespace.strip()
    if namespace in self.namespaces:
        return True
    elif self.validator.meta_schema is None:
        return False
    for schema in self.iter_schemas():
        for url in schema.get_locations(namespace):
            if url in self.missing_locations:
                continue
            try:
                if schema.import_schema(namespace, url, schema.base_url) is not None:
                    if build:
                        self.build()
            except (OSError, IOError):
                pass
            except XMLSchemaNotBuiltError:
                self.clear(remove_schemas=True, only_unbuilt=True)
                self.missing_locations.append(url)
            else:
                return True
    if namespace in self.validator.fallback_locations:
        url = self.validator.fallback_locations[namespace]
        if url not in self.missing_locations:
            try:
                if self.validator.import_schema(namespace, url) is not None:
                    if build:
                        self.build()
            except (OSError, IOError):
                return False
            except XMLSchemaNotBuiltError:
                self.clear(remove_schemas=True, only_unbuilt=True)
                self.missing_locations.append(url)
            else:
                return True
    return False