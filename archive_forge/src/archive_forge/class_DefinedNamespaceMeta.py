import logging
import warnings
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, Tuple, Union
from unicodedata import category
from urllib.parse import urldefrag, urljoin
from rdflib.term import URIRef, Variable, _is_valid_uri
from rdflib.namespace._BRICK import BRICK
from rdflib.namespace._CSVW import CSVW
from rdflib.namespace._DC import DC
from rdflib.namespace._DCAM import DCAM
from rdflib.namespace._DCAT import DCAT
from rdflib.namespace._DCMITYPE import DCMITYPE
from rdflib.namespace._DCTERMS import DCTERMS
from rdflib.namespace._DOAP import DOAP
from rdflib.namespace._FOAF import FOAF
from rdflib.namespace._GEO import GEO
from rdflib.namespace._ODRL2 import ODRL2
from rdflib.namespace._ORG import ORG
from rdflib.namespace._OWL import OWL
from rdflib.namespace._PROF import PROF
from rdflib.namespace._PROV import PROV
from rdflib.namespace._QB import QB
from rdflib.namespace._RDF import RDF
from rdflib.namespace._RDFS import RDFS
from rdflib.namespace._SDO import SDO
from rdflib.namespace._SH import SH
from rdflib.namespace._SKOS import SKOS
from rdflib.namespace._SOSA import SOSA
from rdflib.namespace._SSN import SSN
from rdflib.namespace._TIME import TIME
from rdflib.namespace._VANN import VANN
from rdflib.namespace._VOID import VOID
from rdflib.namespace._WGS import WGS
from rdflib.namespace._XSD import XSD
class DefinedNamespaceMeta(type):
    """Utility metaclass for generating URIRefs with a common prefix."""
    _NS: Namespace
    _warn: bool = True
    _fail: bool = False
    _extras: List[str] = []
    _underscore_num: bool = False

    @lru_cache(maxsize=None)
    def __getitem__(cls, name: str, default=None) -> URIRef:
        name = str(name)
        if name in _DFNS_RESERVED_ATTRS:
            raise AttributeError(f'DefinedNamespace like object has no attribute {name!r}')
        if str(name).startswith('__'):
            return super().__getitem__(name, default)
        if (cls._warn or cls._fail) and name not in cls:
            if cls._fail:
                raise AttributeError(f"term '{name}' not in namespace '{cls._NS}'")
            else:
                warnings.warn(f'Code: {name} is not defined in namespace {cls.__name__}', stacklevel=3)
        return cls._NS[name]

    def __getattr__(cls, name: str):
        return cls.__getitem__(name)

    def __repr__(cls) -> str:
        return f'Namespace({str(cls._NS)!r})'

    def __str__(cls) -> str:
        return str(cls._NS)

    def __add__(cls, other: str) -> URIRef:
        return cls.__getitem__(other)

    def __contains__(cls, item: str) -> bool:
        """Determine whether a URI or an individual item belongs to this namespace"""
        item_str = str(item)
        if item_str.startswith('__'):
            return super().__contains__(item)
        if item_str.startswith(str(cls._NS)):
            item_str = item_str[len(str(cls._NS)):]
        return any((item_str in c.__annotations__ or item_str in c._extras or (cls._underscore_num and item_str[0] == '_' and item_str[1:].isdigit()) for c in cls.mro() if issubclass(c, DefinedNamespace)))

    def __dir__(cls) -> Iterable[str]:
        attrs = {str(x) for x in cls.__annotations__}
        attrs.difference_update(_DFNS_RESERVED_ATTRS)
        values = {cls[str(x)] for x in attrs}
        return values

    def as_jsonld_context(self, pfx: str) -> dict:
        """Returns this DefinedNamespace as a a JSON-LD 'context' object"""
        terms = {pfx: str(self._NS)}
        for key, term in self.__annotations__.items():
            if issubclass(term, URIRef):
                terms[key] = f'{pfx}:{key}'
        return {'@context': terms}