from rdflib.exceptions import Error
from rdflib.namespace import RDF
from rdflib.term import BNode, Literal, URIRef
from .turtle import RecursiveSerializer
def addNamespace(self, prefix, namespace):
    if prefix > '' and prefix[0] == '_' or self.namespaces.get(prefix, namespace) != namespace:
        if prefix not in self._ns_rewrite:
            p = 'p' + prefix
            while p in self.namespaces:
                p = 'p' + p
            self._ns_rewrite[prefix] = p
        prefix = self._ns_rewrite.get(prefix, prefix)
    super(LongTurtleSerializer, self).addNamespace(prefix, namespace)
    return prefix