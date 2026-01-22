from typing import IO, Optional
from rdflib.graph import ConjunctiveGraph, Graph
from rdflib.namespace import Namespace
from rdflib.plugins.serializers.xmlwriter import XMLWriter
from rdflib.serializer import Serializer
from rdflib.term import BNode, Literal, URIRef
def _writeTriple(self, triple):
    self.writer.push(TRIXNS['triple'])
    for component in triple:
        if isinstance(component, URIRef):
            self.writer.element(TRIXNS['uri'], content=str(component))
        elif isinstance(component, BNode):
            self.writer.element(TRIXNS['id'], content=str(component))
        elif isinstance(component, Literal):
            if component.datatype:
                self.writer.element(TRIXNS['typedLiteral'], content=str(component), attributes={TRIXNS['datatype']: str(component.datatype)})
            elif component.language:
                self.writer.element(TRIXNS['plainLiteral'], content=str(component), attributes={XMLNS['lang']: str(component.language)})
            else:
                self.writer.element(TRIXNS['plainLiteral'], content=str(component))
    self.writer.pop()