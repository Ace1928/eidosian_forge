from __future__ import absolute_import, division, print_function
from collections import defaultdict
from datetime import datetime
from hashlib import sha256
from typing import (
from rdflib.graph import ConjunctiveGraph, Graph, ReadOnlyGraphAggregate, _TripleType
from rdflib.term import BNode, IdentifiedNode, Node, URIRef
@_runtime('to_hash_runtime')
def to_hash(self, stats: Optional[Stats]=None):
    result = 0
    for triple in self.canonical_triples(stats=stats):
        result += self.hashfunc(' '.join([x.n3() for x in triple]))
    if stats is not None:
        stats['graph_digest'] = '%x' % result
    return result