from rdflib import BNode
from rdflib import Literal, URIRef
from rdflib import RDF as ns_rdf
from rdflib.term import XSDToPython
from . import IncorrectBlankNodeUsage, IncorrectLiteral, err_no_blank_node
from .utils import has_one_of_attributes, return_XML
import re
def _create_Literal(self, val, datatype='', lang=''):
    """
        Create a literal, taking into account the datatype and language.
        @return: Literal
        """
    if datatype == None or datatype == '':
        return Literal(val, lang=lang)
    else:
        convFunc = XSDToPython.get(datatype, None)
        if convFunc:
            try:
                _pv = convFunc(val)
            except:
                self.state.options.add_warning('Incompatible value (%s) and datatype (%s) in Literal definition.' % (val, datatype), warning_type=IncorrectLiteral, node=self.node.nodeName)
        return Literal(val, datatype=datatype)