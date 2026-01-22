from __future__ import annotations
import collections
import datetime
import itertools
import typing as t
from collections.abc import Mapping, MutableMapping
from typing import (
import isodate
import rdflib.plugins.sparql
from rdflib.graph import ConjunctiveGraph, Graph
from rdflib.namespace import NamespaceManager
from rdflib.plugins.sparql.parserutils import CompValue
from rdflib.term import BNode, Identifier, Literal, Node, URIRef, Variable
class Prologue:
    """
    A class for holding prefixing bindings and base URI information
    """

    def __init__(self) -> None:
        self.base: Optional[str] = None
        self.namespace_manager = NamespaceManager(Graph())

    def resolvePName(self, prefix: Optional[str], localname: Optional[str]) -> URIRef:
        ns = self.namespace_manager.store.namespace(prefix or '')
        if ns is None:
            raise Exception('Unknown namespace prefix : %s' % prefix)
        return URIRef(ns + (localname or ''))

    def bind(self, prefix: Optional[str], uri: Any) -> None:
        self.namespace_manager.bind(prefix, uri, replace=True)

    def absolutize(self, iri: Optional[Union[CompValue, str]]) -> Optional[Union[CompValue, str]]:
        """
        Apply BASE / PREFIXes to URIs
        (and to datatypes in Literals)

        TODO: Move resolving URIs to pre-processing
        """
        if isinstance(iri, CompValue):
            if iri.name == 'pname':
                return self.resolvePName(iri.prefix, iri.localname)
            if iri.name == 'literal':
                return Literal(iri.string, lang=iri.lang, datatype=self.absolutize(iri.datatype))
        elif isinstance(iri, URIRef) and (not ':' in iri):
            return URIRef(iri, base=self.base)
        return iri