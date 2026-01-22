from __future__ import annotations
import codecs
import csv
from typing import IO, Dict, List, Optional, Union
from rdflib.plugins.sparql.processor import SPARQLResult
from rdflib.query import Result, ResultParser, ResultSerializer
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
class CSVResultSerializer(ResultSerializer):

    def __init__(self, result: SPARQLResult):
        ResultSerializer.__init__(self, result)
        self.delim = ','
        if result.type != 'SELECT':
            raise Exception('CSVSerializer can only serialize select query results')

    def serialize(self, stream: IO, encoding: str='utf-8', **kwargs) -> None:
        import codecs
        stream = codecs.getwriter(encoding)(stream)
        out = csv.writer(stream, delimiter=self.delim)
        vs = [self.serializeTerm(v, encoding) for v in self.result.vars]
        out.writerow(vs)
        for row in self.result.bindings:
            out.writerow([self.serializeTerm(row.get(v), encoding) for v in self.result.vars])

    def serializeTerm(self, term: Optional[Identifier], encoding: str) -> Union[str, Identifier]:
        if term is None:
            return ''
        elif isinstance(term, BNode):
            return f'_:{term}'
        else:
            return term