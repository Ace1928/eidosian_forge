from typing import IO, TYPE_CHECKING, Dict, List, Optional, Tuple, Union
from rdflib.graph import ConjunctiveGraph, Graph
from rdflib.plugins.serializers.turtle import TurtleSerializer
from rdflib.term import BNode, Node

Trig RDF graph serializer for RDFLib.
See <http://www.w3.org/TR/trig/> for syntax specification.
