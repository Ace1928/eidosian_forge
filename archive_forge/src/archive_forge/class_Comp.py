from __future__ import annotations
from collections import OrderedDict
from types import MethodType
from typing import (
from pyparsing import ParserElement, ParseResults, TokenConverter, originalTextFor
from rdflib.term import BNode, Identifier, Variable
from rdflib.plugins.sparql.sparql import NotBoundError, SPARQLError  # noqa: E402
class Comp(TokenConverter):
    """
    A pyparsing token for grouping together things with a label
    Any sub-tokens that are not Params will be ignored.

    Returns CompValue / Expr objects - depending on whether evalFn is set.
    """

    def __init__(self, name: str, expr: ParserElement):
        self.expr = expr
        TokenConverter.__init__(self, expr)
        self.setName(name)
        self.evalfn: Optional[Callable[[Any, Any], Any]] = None

    def postParse(self, instring: str, loc: int, tokenList: ParseResults) -> Union[Expr, CompValue]:
        res: Union[Expr, CompValue]
        if self.evalfn:
            res = Expr(self.name)
            res._evalfn = MethodType(self.evalfn, res)
        else:
            res = CompValue(self.name)
            if self.name == 'ServiceGraphPattern':
                sgp = originalTextFor(self.expr)
                service_string = sgp.searchString(instring)[0][0]
                res['service_string'] = service_string
        for t in tokenList:
            if isinstance(t, ParamValue):
                if t.isList:
                    if t.name not in res:
                        res[t.name] = []
                    res[t.name].append(t.tokenList)
                else:
                    res[t.name] = t.tokenList
        return res

    def setEvalFn(self, evalfn: Callable[[Any, Any], Any]) -> Comp:
        self.evalfn = evalfn
        return self