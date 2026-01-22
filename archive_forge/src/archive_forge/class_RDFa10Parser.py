from rdflib.parser import (
from . import pyRdfa, Options
from .embeddedRDF import handle_embeddedRDF
from .state import ExecutionContext
class RDFa10Parser(Parser):
    """
    This is just a convenience class to wrap around the RDFa 1.0 parser.
    """

    def parse(self, source, graph, pgraph=None, media_type=''):
        """
        @param source: one of the input sources that the RDFLib package defined
        @type source: InputSource class instance
        @param graph: target graph for the triples; output graph, in RDFa
        spec. parlance
        @type graph: RDFLib Graph
        @keyword pgraph: target for error and warning triples; processor
        graph, in RDFa spec. parlance. If set to None, these triples are
        ignored
        @type pgraph: RDFLib Graph
        @keyword media_type: explicit setting of the preferred media type
        (a.k.a. content type) of the the RDFa source. None means the content
        type of the HTTP result is used, or a guess is made based on the
        suffix of a file
        @type media_type: string
        @keyword rdfOutput: whether Exceptions should be catched and added,
        as triples, to the processor graph, or whether they should be raised.
        @type rdfOutput: Boolean
        """
        RDFaParser().parse(source, graph, pgraph=pgraph, media_type=media_type, rdfa_version='1.0')