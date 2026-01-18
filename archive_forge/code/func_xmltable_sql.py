from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def xmltable_sql(self, expression: exp.XMLTable) -> str:
    this = self.sql(expression, 'this')
    passing = self.expressions(expression, key='passing')
    passing = f'{self.sep()}PASSING{self.seg(passing)}' if passing else ''
    columns = self.expressions(expression, key='columns')
    columns = f'{self.sep()}COLUMNS{self.seg(columns)}' if columns else ''
    by_ref = f'{self.sep()}RETURNING SEQUENCE BY REF' if expression.args.get('by_ref') else ''
    return f'XMLTABLE({self.sep('')}{self.indent(this + passing + by_ref + columns)}{self.seg(')', sep='')}'