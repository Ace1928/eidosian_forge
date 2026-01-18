import threading
from typing import TYPE_CHECKING, Any, Generator, Iterator, List, Optional, Tuple
from rdflib.graph import ConjunctiveGraph, Graph
from rdflib.store import Store
def triples(self, triple: '_TriplePatternType', context: Optional['_ContextType']=None) -> Iterator[Tuple['_TripleType', Iterator[Optional['_ContextType']]]]:
    su, pr, ob = triple
    context = context.__class__(self.store, context.identifier) if context is not None else None
    for (s, p, o), cg in self.store.triples((su, pr, ob), context):
        yield ((s, p, o), cg)