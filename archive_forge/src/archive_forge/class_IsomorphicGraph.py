from __future__ import absolute_import, division, print_function
from collections import defaultdict
from datetime import datetime
from hashlib import sha256
from typing import (
from rdflib.graph import ConjunctiveGraph, Graph, ReadOnlyGraphAggregate, _TripleType
from rdflib.term import BNode, IdentifiedNode, Node, URIRef
class IsomorphicGraph(ConjunctiveGraph):
    """An implementation of the RGDA1 graph digest algorithm.

    An implementation of RGDA1 (publication below),
    a combination of Sayers & Karp's graph digest algorithm using
    sum and SHA-256 <http://www.hpl.hp.com/techreports/2003/HPL-2003-235R1.pdf>
    and traces <http://pallini.di.uniroma1.it>, an average case
    polynomial time algorithm for graph canonicalization.

    McCusker, J. P. (2015). WebSig: A Digital Signature Framework for the Web.
    Rensselaer Polytechnic Institute, Troy, NY.
    http://gradworks.umi.com/3727015.pdf
    """

    def __init__(self, **kwargs):
        super(IsomorphicGraph, self).__init__(**kwargs)

    def __eq__(self, other):
        """Graph isomorphism testing."""
        if not isinstance(other, IsomorphicGraph):
            return False
        elif len(self) != len(other):
            return False
        return self.internal_hash() == other.internal_hash()

    def __ne__(self, other):
        """Negative graph isomorphism testing."""
        return not self.__eq__(other)

    def __hash__(self):
        return super(IsomorphicGraph, self).__hash__()

    def graph_digest(self, stats=None):
        """Synonym for IsomorphicGraph.internal_hash."""
        return self.internal_hash(stats=stats)

    def internal_hash(self, stats=None):
        """
        This is defined instead of __hash__ to avoid a circular recursion
        scenario with the Memory store for rdflib which requires a hash lookup
        in order to return a generator of triples.
        """
        return _TripleCanonicalizer(self).to_hash(stats=stats)