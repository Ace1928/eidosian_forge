from rdflib.parser import (
from . import pyRdfa, Options
from .embeddedRDF import handle_embeddedRDF
from .state import ExecutionContext
class RDFaParser(Parser):
    """
    Wrapper around the RDFa 1.1 parser. For further details on the RDFa 1.1
    processing, see the relevant W3C documents at
    http://www.w3.org/TR/#tr_RDFa. RDFa 1.1 is defined for XHTML, HTML5, SVG
    and, in general, for any XML language.

    Note that the parser can also handle RDFa 1.0 if the extra parameter is
    used and/or the input source uses RDFa 1.0 specific @version or DTD-s.
    """

    def parse(self, source, graph, pgraph=None, media_type='', rdfa_version=None, embedded_rdf=False, space_preserve=True, vocab_expansion=False, vocab_cache=False, refresh_vocab_cache=False, vocab_cache_report=False, check_lite=False):
        """
        @param source: one of the input sources that the RDFLib package defined
        @type source: InputSource class instance
        @param graph: target graph for the triples; output graph, in RDFa spec.
        parlance
        @type graph: RDFLib Graph
        @keyword pgraph: target for error and warning triples; processor graph,
        in RDFa spec. parlance. If set to None, these triples are ignored
        @type pgraph: RDFLib Graph
        @keyword media_type: explicit setting of the preferred media type
        (a.k.a. content type) of the the RDFa source. None means the content
        type of the HTTP result is used, or a guess is made based on the
        suffix of a file
        @type media_type: string
        @keyword rdfa_version: 1.0 or 1.1. If the value is "", then, by
        default, 1.1 is used unless the source has explicit signals to use
        1.0 (e.g., using a @version attribute, using a DTD set up for 1.0, etc)
        @type rdfa_version: string
        @keyword embedded_rdf: some formats allow embedding RDF in other
        formats: (X)HTML can contain turtle in a special <script> element,
        SVG can have RDF/XML embedded in a <metadata> element. This flag
        controls whether those triples should be interpreted and added to
        the output graph. Some languages (e.g., SVG) require this, and the
        flag is ignored.
        @type embedded_rdf: Boolean
        @keyword space_preserve: by default, space in the HTML source must be preserved in the generated literal;
        this behavior can be switched off
        @type space_preserve: Boolean
        @keyword vocab_expansion: whether the RDFa @vocab attribute should
        also mean vocabulary expansion (see the RDFa 1.1 spec for further
        details)
        @type vocab_expansion: Boolean
        @keyword vocab_cache: in case vocab expansion is used, whether the
        expansion data (i.e., vocabulary) should be cached locally. This
        requires the ability for the local application to write on the
        local file system
        @type vocab_chache: Boolean
        @keyword vocab_cache_report: whether the details of vocabulary file caching process should be reported
        in the processor graph as information (mainly useful for debug)
        @type vocab_cache_report: Boolean
        @keyword refresh_vocab_cache: whether the caching checks of vocabs should be by-passed, ie, if caches should be re-generated regardless of the stored date (important for vocab development)
        @type refresh_vocab_cache: Boolean
        @keyword check_lite: generate extra warnings in case the input source is not RDFa 1.1 check_lite
        @type check_lite: Boolean
        """
        if html5lib is False:
            raise ImportError('html5lib is not installed, cannot use ' + 'RDFa and Microdata parsers.')
        baseURI, orig_source = _get_orig_source(source)
        self._process(graph, pgraph, baseURI, orig_source, media_type=media_type, rdfa_version=rdfa_version, embedded_rdf=embedded_rdf, space_preserve=space_preserve, vocab_expansion=vocab_expansion, vocab_cache=vocab_cache, vocab_cache_report=vocab_cache_report, refresh_vocab_cache=refresh_vocab_cache, check_lite=check_lite)

    def _process(self, graph, pgraph, baseURI, orig_source, media_type='', rdfa_version=None, embedded_rdf=False, space_preserve=True, vocab_expansion=False, vocab_cache=False, vocab_cache_report=False, refresh_vocab_cache=False, check_lite=False):
        from rdflib import Graph
        processor_graph = pgraph if pgraph is not None else Graph()
        self.options = Options(output_processor_graph=True, embedded_rdf=embedded_rdf, space_preserve=space_preserve, vocab_expansion=vocab_expansion, vocab_cache=vocab_cache, vocab_cache_report=vocab_cache_report, refresh_vocab_cache=refresh_vocab_cache, check_lite=check_lite)
        if media_type is None:
            media_type = ''
        processor = pyRdfa(self.options, base=baseURI, media_type=media_type, rdfa_version=rdfa_version)
        processor.graph_from_source(orig_source, graph=graph, pgraph=processor_graph, rdfOutput=False)
        _check_error(processor_graph)