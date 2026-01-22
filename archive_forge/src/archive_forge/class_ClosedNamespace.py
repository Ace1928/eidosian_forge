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
class ClosedNamespace(Namespace):
    """
    A namespace with a closed list of members

    Trying to create terms not listed is an error
    """
    __uris: Dict[str, URIRef]

    def __new__(cls, uri: str, terms: List[str]):
        rt = super().__new__(cls, uri)
        rt.__uris = {t: URIRef(rt + t) for t in terms}
        return rt

    @property
    def uri(self) -> str:
        return str(self)

    def term(self, name: str) -> URIRef:
        uri = self.__uris.get(name)
        if uri is None:
            raise KeyError(f"term '{name}' not in namespace '{self}'")
        return uri

    def __getitem__(self, key: str) -> URIRef:
        return self.term(key)

    def __getattr__(self, name: str) -> URIRef:
        if name.startswith('__'):
            raise AttributeError
        else:
            try:
                return self.term(name)
            except KeyError as e:
                raise AttributeError(e)

    def __repr__(self) -> str:
        return f'{self.__module__}.{self.__class__.__name__}({str(self)!r})'

    def __dir__(self) -> List[str]:
        return list(self.__uris)

    def __contains__(self, ref: str) -> bool:
        return ref in self.__uris.values()

    def _ipython_key_completions_(self) -> List[str]:
        return dir(self)