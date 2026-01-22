from random import randint
from rdflib.namespace import RDF
from rdflib.term import BNode, URIRef
class NoElementException(Exception):

    def __init__(self, message='rdf:Alt Container is empty'):
        self.message = message

    def __str__(self):
        return self.message