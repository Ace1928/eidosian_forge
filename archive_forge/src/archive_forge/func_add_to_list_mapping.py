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
def add_to_list_mapping(self, prop, resource):
    """Add a new property-resource on the list mapping structure. The latter is a dictionary of arrays;
        if the array does not exist yet, it will be created on the fly.
        
        @param prop: the property URI, used as a key in the dictionary
        @param resource: the resource to be added to the relevant array in the dictionary. Can be None; this is a dummy
        placeholder for C{<span rel="property" inlist>...</span>} constructions that may be filled in by children or siblings; if not
        an empty list has to be generated.
        """
    if prop in self.list_mapping.mapping:
        if resource != None:
            if self.list_mapping.mapping[prop] == None:
                self.list_mapping.mapping[prop] = [resource]
            else:
                self.list_mapping.mapping[prop].append(resource)
    elif resource != None:
        self.list_mapping.mapping[prop] = [resource]
    else:
        self.list_mapping.mapping[prop] = None