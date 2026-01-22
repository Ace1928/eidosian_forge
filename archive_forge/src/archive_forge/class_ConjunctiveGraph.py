from __future__ import annotations
import logging
import pathlib
import random
from io import BytesIO
from typing import (
from urllib.parse import urlparse
from urllib.request import url2pathname
import rdflib.exceptions as exceptions
import rdflib.namespace as namespace  # noqa: F401 # This is here because it is used in a docstring.
import rdflib.plugin as plugin
import rdflib.query as query
import rdflib.util  # avoid circular dependency
from rdflib.collection import Collection
from rdflib.exceptions import ParserError
from rdflib.namespace import RDF, Namespace, NamespaceManager
from rdflib.parser import InputSource, Parser, create_input_source
from rdflib.paths import Path
from rdflib.resource import Resource
from rdflib.serializer import Serializer
from rdflib.store import Store
from rdflib.term import (
class ConjunctiveGraph(Graph):
    """A ConjunctiveGraph is an (unnamed) aggregation of all the named
    graphs in a store.

    It has a ``default`` graph, whose name is associated with the
    graph throughout its life. :meth:`__init__` can take an identifier
    to use as the name of this default graph or it will assign a
    BNode.

    All methods that add triples work against this default graph.

    All queries are carried out against the union of all graphs.
    """

    def __init__(self, store: Union[Store, str]='default', identifier: Optional[Union[IdentifiedNode, str]]=None, default_graph_base: Optional[str]=None):
        super(ConjunctiveGraph, self).__init__(store, identifier=identifier)
        assert self.store.context_aware, 'ConjunctiveGraph must be backed by a context aware store.'
        self.context_aware = True
        self.default_union = True
        self.default_context: _ContextType = Graph(store=self.store, identifier=identifier or BNode(), base=default_graph_base)

    def __str__(self) -> str:
        pattern = "[a rdflib:ConjunctiveGraph;rdflib:storage [a rdflib:Store;rdfs:label '%s']]"
        return pattern % self.store.__class__.__name__

    @overload
    def _spoc(self, triple_or_quad: '_QuadType', default: bool=False) -> '_QuadType':
        ...

    @overload
    def _spoc(self, triple_or_quad: Union['_TripleType', '_OptionalQuadType'], default: bool=False) -> '_OptionalQuadType':
        ...

    @overload
    def _spoc(self, triple_or_quad: None, default: bool=False) -> Tuple[None, None, None, Optional[Graph]]:
        ...

    @overload
    def _spoc(self, triple_or_quad: Optional[_TripleOrQuadPatternType], default: bool=False) -> '_QuadPatternType':
        ...

    @overload
    def _spoc(self, triple_or_quad: _TripleOrQuadSelectorType, default: bool=False) -> _QuadSelectorType:
        ...

    @overload
    def _spoc(self, triple_or_quad: Optional[_TripleOrQuadSelectorType], default: bool=False) -> _QuadSelectorType:
        ...

    def _spoc(self, triple_or_quad: Optional[_TripleOrQuadSelectorType], default: bool=False) -> _QuadSelectorType:
        """
        helper method for having methods that support
        either triples or quads
        """
        if triple_or_quad is None:
            return (None, None, None, self.default_context if default else None)
        if len(triple_or_quad) == 3:
            c = self.default_context if default else None
            s, p, o = triple_or_quad
        elif len(triple_or_quad) == 4:
            s, p, o, c = triple_or_quad
            c = self._graph(c)
        return (s, p, o, c)

    def __contains__(self, triple_or_quad: _TripleOrQuadSelectorType) -> bool:
        """Support for 'triple/quad in graph' syntax"""
        s, p, o, c = self._spoc(triple_or_quad)
        for t in self.triples((s, p, o), context=c):
            return True
        return False

    def add(self: _ConjunctiveGraphT, triple_or_quad: _TripleOrOptionalQuadType) -> _ConjunctiveGraphT:
        """
        Add a triple or quad to the store.

        if a triple is given it is added to the default context
        """
        s, p, o, c = self._spoc(triple_or_quad, default=True)
        _assertnode(s, p, o)
        self.store.add((s, p, o), context=c, quoted=False)
        return self

    @overload
    def _graph(self, c: Union[Graph, _ContextIdentifierType, str]) -> Graph:
        ...

    @overload
    def _graph(self, c: None) -> None:
        ...

    def _graph(self, c: Optional[Union[Graph, _ContextIdentifierType, str]]) -> Optional[Graph]:
        if c is None:
            return None
        if not isinstance(c, Graph):
            return self.get_context(c)
        else:
            return c

    def addN(self: _ConjunctiveGraphT, quads: Iterable['_QuadType']) -> _ConjunctiveGraphT:
        """Add a sequence of triples with context"""
        self.store.addN(((s, p, o, self._graph(c)) for s, p, o, c in quads if _assertnode(s, p, o)))
        return self

    def remove(self: _ConjunctiveGraphT, triple_or_quad: _TripleOrOptionalQuadType) -> _ConjunctiveGraphT:
        """
        Removes a triple or quads

        if a triple is given it is removed from all contexts

        a quad is removed from the given context only

        """
        s, p, o, c = self._spoc(triple_or_quad)
        self.store.remove((s, p, o), context=c)
        return self

    @overload
    def triples(self, triple_or_quad: '_TripleOrQuadPatternType', context: Optional[_ContextType]=...) -> Generator['_TripleType', None, None]:
        ...

    @overload
    def triples(self, triple_or_quad: '_TripleOrQuadPathPatternType', context: Optional[_ContextType]=...) -> Generator['_TriplePathType', None, None]:
        ...

    @overload
    def triples(self, triple_or_quad: _TripleOrQuadSelectorType, context: Optional[_ContextType]=...) -> Generator['_TripleOrTriplePathType', None, None]:
        ...

    def triples(self, triple_or_quad: _TripleOrQuadSelectorType, context: Optional[_ContextType]=None) -> Generator['_TripleOrTriplePathType', None, None]:
        """
        Iterate over all the triples in the entire conjunctive graph

        For legacy reasons, this can take the context to query either
        as a fourth element of the quad, or as the explicit context
        keyword parameter. The kw param takes precedence.
        """
        s, p, o, c = self._spoc(triple_or_quad)
        context = self._graph(context or c)
        if self.default_union:
            if context == self.default_context:
                context = None
        elif context is None:
            context = self.default_context
        if isinstance(p, Path):
            if context is None:
                context = self
            for s, o in p.eval(context, s, o):
                yield (s, p, o)
        else:
            for (s, p, o), cg in self.store.triples((s, p, o), context=context):
                yield (s, p, o)

    def quads(self, triple_or_quad: Optional[_TripleOrQuadPatternType]=None) -> Generator[_OptionalQuadType, None, None]:
        """Iterate over all the quads in the entire conjunctive graph"""
        s, p, o, c = self._spoc(triple_or_quad)
        for (s, p, o), cg in self.store.triples((s, p, o), context=c):
            for ctx in cg:
                yield (s, p, o, ctx)

    def triples_choices(self, triple: Union[Tuple[List['_SubjectType'], '_PredicateType', '_ObjectType'], Tuple['_SubjectType', List['_PredicateType'], '_ObjectType'], Tuple['_SubjectType', '_PredicateType', List['_ObjectType']]], context: Optional['_ContextType']=None) -> Generator[_TripleType, None, None]:
        """Iterate over all the triples in the entire conjunctive graph"""
        s, p, o = triple
        if context is None:
            if not self.default_union:
                context = self.default_context
        else:
            context = self._graph(context)
        for (s1, p1, o1), cg in self.store.triples_choices((s, p, o), context=context):
            yield (s1, p1, o1)

    def __len__(self) -> int:
        """Number of triples in the entire conjunctive graph"""
        return self.store.__len__()

    def contexts(self, triple: Optional['_TripleType']=None) -> Generator['_ContextType', None, None]:
        """Iterate over all contexts in the graph

        If triple is specified, iterate over all contexts the triple is in.
        """
        for context in self.store.contexts(triple):
            if isinstance(context, Graph):
                yield context
            else:
                yield self.get_context(context)

    def get_graph(self, identifier: '_ContextIdentifierType') -> Union[Graph, None]:
        """Returns the graph identified by given identifier"""
        return [x for x in self.contexts() if x.identifier == identifier][0]

    def get_context(self, identifier: Optional[Union['_ContextIdentifierType', str]], quoted: bool=False, base: Optional[str]=None) -> Graph:
        """Return a context graph for the given identifier

        identifier must be a URIRef or BNode.
        """
        return Graph(store=self.store, identifier=identifier, namespace_manager=self.namespace_manager, base=base)

    def remove_context(self, context: '_ContextType') -> None:
        """Removes the given context from the graph"""
        self.store.remove((None, None, None), context)

    def context_id(self, uri: str, context_id: Optional[str]=None) -> URIRef:
        """URI#context"""
        uri = uri.split('#', 1)[0]
        if context_id is None:
            context_id = '#context'
        return URIRef(context_id, base=uri)

    def parse(self, source: Optional[Union[IO[bytes], TextIO, InputSource, str, bytes, pathlib.PurePath]]=None, publicID: Optional[str]=None, format: Optional[str]=None, location: Optional[str]=None, file: Optional[Union[BinaryIO, TextIO]]=None, data: Optional[Union[str, bytes]]=None, **args: Any) -> 'Graph':
        """
        Parse source adding the resulting triples to its own context (sub graph
        of this graph).

        See :meth:`rdflib.graph.Graph.parse` for documentation on arguments.

        If the source is in a format that does not support named graphs it's triples
        will be added to the default graph (i.e. `Dataset.default_context`).

        :Returns:

        The graph into which the source was parsed. In the case of n3 it returns
        the root context.

        .. caution::

           This method can access directly or indirectly requested network or
           file resources, for example, when parsing JSON-LD documents with
           ``@context`` directives that point to a network location.

           When processing untrusted or potentially malicious documents,
           measures should be taken to restrict network and file access.

           For information on available security measures, see the RDFLib
           :doc:`Security Considerations </security_considerations>`
           documentation.

        *Changed in 7.0*: The ``publicID`` argument is no longer used as the
        identifier (i.e. name) of the default graph as was the case before
        version 7.0. In the case of sources that do not support named graphs,
        the ``publicID`` parameter will also not be used as the name for the
        graph that the data is loaded into, and instead the triples from sources
        that do not support named graphs will be loaded into the default graph
        (i.e. `ConjunctionGraph.default_context`).
        """
        source = create_input_source(source=source, publicID=publicID, location=location, file=file, data=data, format=format)
        context = self.default_context
        context.parse(source, publicID=publicID, format=format, **args)
        return context

    def __reduce__(self) -> Tuple[Type[Graph], Tuple[Store, _ContextIdentifierType]]:
        return (ConjunctiveGraph, (self.store, self.identifier))