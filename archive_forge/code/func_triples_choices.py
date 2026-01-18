import collections
import re
from typing import (
from rdflib.graph import DATASET_DEFAULT_GRAPH_ID, Graph
from rdflib.plugins.stores.regexmatching import NATIVE_REGEX
from rdflib.store import Store
from rdflib.term import BNode, Identifier, Node, URIRef, Variable
from .sparqlconnector import SPARQLConnector
def triples_choices(self, _: Tuple[Union['_SubjectType', List['_SubjectType']], Union['_PredicateType', List['_PredicateType']], Union['_ObjectType', List['_ObjectType']]], context: Optional['_ContextType']=None) -> Generator[Tuple[Tuple['_SubjectType', '_PredicateType', '_ObjectType'], Iterator[Optional['_ContextType']]], None, None]:
    """
        A variant of triples that can take a list of terms instead of a
        single term in any slot.  Stores can implement this to optimize
        the response time from the import default 'fallback' implementation,
        which will iterate over each term in the list and dispatch to
        triples.
        """
    raise NotImplementedError('Triples choices currently not supported')