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
class URIPattern(str):
    """
    Utility class for creating URIs according to some pattern
    This supports either new style formatting with .format
    or old-style with % operator

    >>> u=URIPattern("http://example.org/%s/%d/resource")
    >>> u%('books', 12345)
    rdflib.term.URIRef('http://example.org/books/12345/resource')

    """

    def __new__(cls, value: Union[str, bytes]) -> 'URIPattern':
        try:
            rt = str.__new__(cls, value)
        except UnicodeDecodeError:
            if TYPE_CHECKING:
                assert isinstance(value, bytes)
            rt = str.__new__(cls, value, 'utf-8')
        return rt

    def __mod__(self, *args, **kwargs) -> URIRef:
        return URIRef(super().__mod__(*args, **kwargs))

    def format(self, *args, **kwargs) -> URIRef:
        return URIRef(super().format(*args, **kwargs))

    def __repr__(self) -> str:
        return f'URIPattern({super().__repr__()})'