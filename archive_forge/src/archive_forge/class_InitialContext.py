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
class InitialContext:
    """
    Get the initial context values. In most cases this class has an empty content, except for the
    top level (in case of RDFa 1.1). Each L{TermOrCurie} class has one instance of this class. It provides initial
    mappings for terms, namespace prefixes, etc, that the top level L{TermOrCurie} instance uses for its own initialization.
    
    @ivar terms: collection of all term mappings
    @type terms: dictionary
    @ivar ns: namespace mapping
    @type ns: dictionary
    @ivar vocabulary: default vocabulary
    @type vocabulary: string
    """

    def __init__(self, state, top_level):
        """
        @param state: the state behind this term mapping
        @type state: L{state.ExecutionContext}
        @param top_level : whether this is the top node of the DOM tree (the only place where initial contexts are handled)
        @type top_level : boolean
        """
        self.state = state
        self.terms = {}
        self.ns = {}
        self.vocabulary = None
        if state.rdfa_version < '1.1' or top_level == False:
            return
        from .initialcontext import initial_context as context_data
        from .host import initial_contexts as context_ids
        from .host import default_vocabulary
        for i in context_ids[state.options.host_language]:
            data = context_data[i]
            if state.options.host_language in default_vocabulary:
                self.vocabulary = default_vocabulary[state.options.host_language]
            elif data.vocabulary != '':
                self.vocabulary = data.vocabulary
            for key in data.terms:
                self.terms[key] = URIRef(data.terms[key])
            for key in data.ns:
                self.ns[key] = (Namespace(data.ns[key]), False)