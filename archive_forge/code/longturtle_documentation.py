from rdflib.exceptions import Error
from rdflib.namespace import RDF
from rdflib.term import BNode, Literal, URIRef
from .turtle import RecursiveSerializer

        Checks if l is a valid RDF list, i.e. no nodes have other properties.
        