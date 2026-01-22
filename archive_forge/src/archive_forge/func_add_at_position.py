from random import randint
from rdflib.namespace import RDF
from rdflib.term import BNode, URIRef
def add_at_position(self, pos, item):
    assert isinstance(pos, int)
    if pos <= 0 or pos > len(self) + 1:
        raise ValueError('Invalid Position for inserting element in rdf:Seq')
    if pos == len(self) + 1:
        self.append(item)
    else:
        for j in range(len(self), pos - 1, -1):
            container = self._get_container()
            elem_uri = str(RDF) + '_' + str(j)
            v = self.graph.value(container, URIRef(elem_uri))
            self.graph.remove((container, URIRef(elem_uri), v))
            elem_uri = str(RDF) + '_' + str(j + 1)
            self.graph.add((container, URIRef(elem_uri), v))
        elem_uri_pos = str(RDF) + '_' + str(pos)
        self.graph.add((container, URIRef(elem_uri_pos), item))
        self._len += 1
    return self