from rdflib import BNode
from rdflib import Literal, URIRef
from rdflib import RDF as ns_rdf
from rdflib.term import XSDToPython
from . import IncorrectBlankNodeUsage, IncorrectLiteral, err_no_blank_node
from .utils import has_one_of_attributes, return_XML
import re
def _get_HTML_literal(self, Pnode):
    """
        Get (recursively) the XML Literal content of a DOM Node. 
    
        @param Pnode: DOM Node
        @return: string
        """
    rc = ''
    for node in Pnode.childNodes:
        if node.nodeType == node.TEXT_NODE:
            rc = rc + self._putBackEntities(node.data)
        elif node.nodeType == node.ELEMENT_NODE:
            rc = rc + return_XML(self.state, node, base=False, xmlns=False)
    return rc