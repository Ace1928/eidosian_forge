import collections
import itertools
import sys
import rdflib.extras.cmdlineutils
from rdflib import RDF, RDFS, XSD

    Convert the RDFS schema in a graph
    writes the dot output to the stream
    