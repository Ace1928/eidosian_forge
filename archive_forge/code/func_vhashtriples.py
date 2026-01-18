from itertools import combinations
from rdflib import BNode, Graph
def vhashtriples(self, term, done):
    for t in self:
        if term in t:
            yield tuple(self.vhashtriple(t, term, done))