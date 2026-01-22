from rdflib.parser import (
from . import pyRdfa, Options
from .embeddedRDF import handle_embeddedRDF
from .state import ExecutionContext
def _process_one_node(node, graph, state):
    if handle_embeddedRDF(node, graph, state):
        return
    else:
        for n in node.childNodes:
            if n.nodeType == node.ELEMENT_NODE:
                _process_one_node(n, graph, state)