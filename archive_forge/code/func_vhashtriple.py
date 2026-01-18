from itertools import combinations
from rdflib import BNode, Graph
def vhashtriple(self, triple, term, done):
    for p in range(3):
        if not isinstance(triple[p], BNode):
            yield triple[p]
        elif done or triple[p] == term:
            yield p
        else:
            yield self.vhash(triple[p], done=True)