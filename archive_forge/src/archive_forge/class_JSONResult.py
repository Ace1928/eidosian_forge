from __future__ import annotations
import json
from typing import IO, Any, Dict, Mapping, MutableSequence, Optional
from rdflib.query import Result, ResultException, ResultParser, ResultSerializer
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
class JSONResult(Result):

    def __init__(self, json: Dict[str, Any]):
        self.json = json
        if 'boolean' in json:
            type_ = 'ASK'
        elif 'results' in json:
            type_ = 'SELECT'
        else:
            raise ResultException('No boolean or results in json!')
        Result.__init__(self, type_)
        if type_ == 'ASK':
            self.askAnswer = bool(json['boolean'])
        else:
            self.bindings = self._get_bindings()
            self.vars = [Variable(x) for x in json['head']['vars']]

    def _get_bindings(self) -> MutableSequence[Mapping[Variable, Identifier]]:
        ret: MutableSequence[Mapping[Variable, Identifier]] = []
        for row in self.json['results']['bindings']:
            outRow: Dict[Variable, Identifier] = {}
            for k, v in row.items():
                outRow[Variable(k)] = parseJsonTerm(v)
            ret.append(outRow)
        return ret