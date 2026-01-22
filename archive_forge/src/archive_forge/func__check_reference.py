import re
from urllib.parse import urlsplit
from rdflib import URIRef
from rdflib import BNode
from rdflib import Namespace
from .utils import quote_URI
from .host import predefined_1_0_rel, warn_xmlns_usage
from . import IncorrectPrefixDefinition, RDFA_VOCAB, UnresolvableReference, PrefixRedefinitionWarning
from . import err_redefining_URI_as_prefix
from . import err_xmlns_deprecated
from . import err_bnode_local_prefix
from . import err_col_local_prefix
from . import err_missing_URI_prefix
from . import err_invalid_prefix
from . import err_no_default_prefix
from . import err_prefix_and_xmlns
from . import err_non_ncname_prefix
from . import err_absolute_reference
from . import err_query_reference
from . import err_fragment_reference
from . import err_prefix_redefinition
def _check_reference(self, val):
    """Checking the CURIE reference for correctness. It is probably not 100% foolproof, but may take care
        of some of the possible errors. See the URI RFC for the details.
        """

    def char_check(s, not_allowed=['#', '[', ']']):
        for c in not_allowed:
            if s.find(c) != -1:
                return False
        return True
    _scheme, netloc, _url, query, fragment = urlsplit('http:' + val)
    if netloc != '' and self.state.rdfa_version >= '1.1':
        self.state.options.add_warning(err_absolute_reference % (netloc, val), UnresolvableReference, node=self.state.node.nodeName)
        return False
    elif not char_check(query):
        self.state.options.add_warning(err_query_reference % (query, val), UnresolvableReference, node=self.state.node.nodeName)
        return False
    elif not char_check(fragment):
        self.state.options.add_warning(err_fragment_reference % (fragment, val), UnresolvableReference, node=self.state.node.nodeName)
        return False
    else:
        return True