from random import randint
from rdflib.namespace import RDF
from rdflib.term import BNode, URIRef
def anyone(self):
    if len(self) == 0:
        raise NoElementException()
    else:
        p = randint(1, len(self))
        item = self.__getitem__(p)
        return item