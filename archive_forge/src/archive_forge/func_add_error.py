import os, sys, datetime, re
from rdflib import Graph
from ..utils import create_file_name
from . import VocabCachingInfo
import pickle
def add_error(self, txt, err_type=None, context=None):
    """Add an error  to the processor graph.
            @param txt: the information text. 
            @keyword err_type: Error Class
            @type err_type: URIRef
            @keyword context: possible context to be added to the processor graph
            @type context: URIRef or String
            """
    self.pr('Error', txt, err_type, context)