from rdflib.parser import (
from . import pyRdfa, Options
from .embeddedRDF import handle_embeddedRDF
from .state import ExecutionContext
class HTurtleParser(Parser):

    def parse(self, source, graph, pgraph=None, media_type=''):
        """
        @param source: one of the input sources that the RDFLib package defined
        @type source: InputSource class instance
        @param graph: target graph for the triples; output graph, in RDFa spec.
        parlance
        @type graph: RDFLib Graph
        @keyword media_type: explicit setting of the preferred media type
        (a.k.a. content type) of the the RDFa source. None means the content
        type of the HTTP result is used, or a guess is made based on the
        suffix of a file
        @type media_type: string
        """
        if html5lib is False:
            raise ImportError('html5lib is not installed, cannot ' + 'use RDFa and Microdata parsers.')
        baseURI, orig_source = _get_orig_source(source)
        self._process(graph, pgraph, baseURI, orig_source, media_type=media_type)

    def _process(self, graph, baseURI, orig_source, media_type=''):
        self.options = Options(output_processor_graph=None, embedded_rdf=True, vocab_expansion=False, vocab_cache=False)
        if media_type is None:
            media_type = ''
        processor = HTurtle(self.options, base=baseURI, media_type=media_type)
        processor.graph_from_source(orig_source, graph=graph, pgraph=None, rdfOutput=False)
        _check_error(graph)