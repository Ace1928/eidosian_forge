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
def _get_plugin_names(kind):
    return ', '.join((p.name for p in plugin.plugins(kind=kind)))