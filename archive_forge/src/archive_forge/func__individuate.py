from __future__ import absolute_import, division, print_function
from collections import defaultdict
from datetime import datetime
from hashlib import sha256
from typing import (
from rdflib.graph import ConjunctiveGraph, Graph, ReadOnlyGraphAggregate, _TripleType
from rdflib.term import BNode, IdentifiedNode, Node, URIRef
def _individuate(self, color, individual):
    new_color = list(color.color)
    new_color.append((len(color.nodes),))
    color.nodes.remove(individual)
    c = Color([individual], self.hashfunc, tuple(new_color), hash_cache=self._hash_cache)
    return c