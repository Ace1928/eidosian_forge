import xml.dom.minidom
from typing import IO, Dict, Optional, Set
from xml.sax.saxutils import escape, quoteattr
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import RDF, RDFS, Namespace  # , split_uri
from rdflib.plugins.parsers.RDFVOC import RDFVOC
from rdflib.plugins.serializers.xmlwriter import XMLWriter
from rdflib.serializer import Serializer
from rdflib.term import BNode, IdentifiedNode, Identifier, Literal, Node, URIRef
from rdflib.util import first, more_than
from .xmlwriter import ESCAPE_ENTITIES
strip off _: from nodeIDs... as they are not valid NCNames