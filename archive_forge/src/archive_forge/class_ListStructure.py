from rdflib import URIRef
from rdflib import BNode
from .host import HostLanguage, accept_xml_base, accept_xml_lang, beautifying_prefixes
from .termorcurie import TermOrCurie
from . import UnresolvablePrefix, UnresolvableTerm
from . import err_URI_scheme
from . import err_illegal_safe_CURIE
from . import err_no_CURIE_in_safe_CURIE
from . import err_undefined_terms
from . import err_non_legal_CURIE_ref
from . import err_undefined_CURIE
from urllib.parse import urlparse, urlunparse, urlsplit, urljoin
class ListStructure:
    """Special class to handle the C{@inlist} type structures in RDFa 1.1; stores the "origin", i.e,
    where the list will be attached to, and the mappings as defined in the spec.
    """

    def __init__(self):
        self.mapping = {}
        self.origin = None