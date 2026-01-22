from __future__ import annotations
import codecs
import os
import re
import sys
import typing
from decimal import Decimal
from typing import (
from uuid import uuid4
from rdflib.compat import long_type
from rdflib.exceptions import ParserError
from rdflib.graph import ConjunctiveGraph, Graph, QuotedGraph
from rdflib.term import (
from rdflib.parser import Parser
class RDFSink:

    def __init__(self, graph: Graph):
        self.rootFormula: Optional[Formula] = None
        self.uuid = uuid4().hex
        self.counter = 0
        self.graph = graph

    def newFormula(self) -> Formula:
        fa = getattr(self.graph.store, 'formula_aware', False)
        if not fa:
            raise ParserError('Cannot create formula parser with non-formula-aware store.')
        f = Formula(self.graph)
        return f

    def newGraph(self, identifier: Identifier) -> Graph:
        return Graph(self.graph.store, identifier)

    def newSymbol(self, *args: str) -> URIRef:
        return URIRef(args[0])

    def newBlankNode(self, arg: Optional[Union[Formula, Graph, Any]]=None, uri: Optional[str]=None, why: Optional[Callable[[], None]]=None) -> BNode:
        if isinstance(arg, Formula):
            return arg.newBlankNode(uri)
        elif isinstance(arg, Graph) or arg is None:
            self.counter += 1
            bn = BNode('n%sb%s' % (self.uuid, self.counter))
        else:
            bn = BNode(str(arg[0]).split('#').pop().replace('_', 'b'))
        return bn

    def newLiteral(self, s: str, dt: Optional[URIRef], lang: Optional[str]) -> Literal:
        if dt:
            return Literal(s, datatype=dt)
        else:
            return Literal(s, lang=lang)

    def newList(self, n: typing.List[Any], f: Optional[Formula]) -> IdentifiedNode:
        nil = self.newSymbol('http://www.w3.org/1999/02/22-rdf-syntax-ns#nil')
        if not n:
            return nil
        first = self.newSymbol('http://www.w3.org/1999/02/22-rdf-syntax-ns#first')
        rest = self.newSymbol('http://www.w3.org/1999/02/22-rdf-syntax-ns#rest')
        af = a = self.newBlankNode(f)
        for ne in n[:-1]:
            self.makeStatement((f, first, a, ne))
            an = self.newBlankNode(f)
            self.makeStatement((f, rest, a, an))
            a = an
        self.makeStatement((f, first, a, n[-1]))
        self.makeStatement((f, rest, a, nil))
        return af

    def newSet(self, *args: _AnyT) -> Set[_AnyT]:
        return set(args)

    def setDefaultNamespace(self, *args: bytes) -> str:
        return ':'.join((repr(n) for n in args))

    def makeStatement(self, quadruple: Tuple[Optional[Union[Formula, Graph]], Node, Node, Node], why: Optional[Any]=None) -> None:
        f, p, s, o = quadruple
        if hasattr(p, 'formula'):
            raise ParserError('Formula used as predicate')
        s = self.normalise(f, s)
        p = self.normalise(f, p)
        o = self.normalise(f, o)
        if f == self.rootFormula:
            self.graph.add((s, p, o))
        elif isinstance(f, Formula):
            f.quotedgraph.add((s, p, o))
        else:
            f.add((s, p, o))

    def normalise(self, f: Optional[Formula], n: Union[Tuple[int, str], bool, int, Decimal, float, _AnyT]) -> Union[URIRef, Literal, BNode, _AnyT]:
        if isinstance(n, tuple):
            return URIRef(str(n[1]))
        if isinstance(n, bool):
            s = Literal(str(n).lower(), datatype=BOOLEAN_DATATYPE)
            return s
        if isinstance(n, int) or isinstance(n, long_type):
            s = Literal(str(n), datatype=INTEGER_DATATYPE)
            return s
        if isinstance(n, Decimal):
            value = str(n)
            if value == '-0':
                value = '0'
            s = Literal(value, datatype=DECIMAL_DATATYPE)
            return s
        if isinstance(n, float):
            s = Literal(str(n), datatype=DOUBLE_DATATYPE)
            return s
        if isinstance(f, Formula):
            if n in f.existentials:
                if TYPE_CHECKING:
                    assert isinstance(n, URIRef)
                return f.existentials[n]
        return n

    def intern(self, something: _AnyT) -> _AnyT:
        return something

    def bind(self, pfx, uri) -> None:
        pass

    def startDoc(self, formula: Optional[Formula]) -> None:
        self.rootFormula = formula

    def endDoc(self, formula: Optional[Formula]) -> None:
        pass