import itertools
import logging
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import OWL, RDF, RDFS, XSD, Namespace, NamespaceManager
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
from rdflib.util import first


        OWL properties map to ACE transitive verbs (TV)

        There are 6 morphological categories that determine the surface form
        of an IRI:

            singular form of a transitive verb (e.g. mans)
            plural form of a transitive verb (e.g. man)
            past participle form a transitive verb (e.g. manned)

            http://attempto.ifi.uzh.ch/ace_lexicon#TV_sg
            http://attempto.ifi.uzh.ch/ace_lexicon#TV_pl
            http://attempto.ifi.uzh.ch/ace_lexicon#TV_vbg

        