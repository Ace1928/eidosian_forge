import datetime
from rdflib import URIRef
from rdflib import Literal
from rdflib import BNode
from rdflib import Namespace
from rdflib import Graph
from rdflib import RDF as ns_rdf
from .host import HostLanguage, content_to_host_language, predefined_1_0_rel, require_embedded_rdf
from . import ns_xsd, ns_distill, ns_rdfa
from . import RDFA_Error, RDFA_Warning, RDFA_Info
from .transform.lite import lite_prune
def add_http_context(self, subj, http_code):
    """
        Add an additional HTTP context to a message with subject in C{subj}, using the U{<http://www.w3.org/2006/http#>}
        vocabulary. Typically used to extend an error structure, as created by L{add_triples}.
        
        @param subj: an RDFLib resource, typically a blank node
        @param http_code: HTTP status code
        """
    bnode = BNode()
    self.graph.add((subj, ns_rdfa['context'], bnode))
    self.graph.add((bnode, ns_rdf['type'], ns_ht['Response']))
    self.graph.add((bnode, ns_ht['responseCode'], URIRef('http://www.w3.org/2006/http#%s' % http_code)))