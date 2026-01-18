import logging
import sys
from optparse import OptionParser
import rdflib
from rdflib import plugin
from rdflib.graph import ConjunctiveGraph
from rdflib.parser import Parser
from rdflib.serializer import Serializer
from rdflib.store import Store
from rdflib.util import guess_format

    >>> _format_and_kws("fmt")
    ('fmt', {})
    >>> _format_and_kws("fmt:+a")
    ('fmt', {'a': True})
    >>> _format_and_kws("fmt:a")
    ('fmt', {'a': True})
    >>> _format_and_kws("fmt:+a,-b") #doctest: +SKIP
    ('fmt', {'a': True, 'b': False})
    >>> _format_and_kws("fmt:c=d")
    ('fmt', {'c': 'd'})
    >>> _format_and_kws("fmt:a=b:c")
    ('fmt', {'a': 'b:c'})
    