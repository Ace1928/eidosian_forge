from __future__ import annotations
import collections
import functools
import operator
import typing
from functools import reduce
from typing import (
from pyparsing import ParseResults
from rdflib.paths import (
from rdflib.plugins.sparql.operators import TrueFilter, and_
from rdflib.plugins.sparql.operators import simplify as simplifyFilters
from rdflib.plugins.sparql.parserutils import CompValue, Expr
from rdflib.plugins.sparql.sparql import Prologue, Query, Update
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
class _AlgebraTranslator:
    """
    Translator of a Query's algebra to its equivalent SPARQL (string).

    Coded as a class to support storage of state during the translation process,
    without use of a file.

    Anticipated Usage:

    .. code-block:: python

        translated_query = _AlgebraTranslator(query).translateAlgebra()

    An external convenience function which wraps the above call,
    `translateAlgebra`, is supplied, so this class does not need to be
    referenced by client code at all in normal use.
    """

    def __init__(self, query_algebra: Query):
        self.query_algebra = query_algebra
        self.aggr_vars: DefaultDict[Identifier, List[Identifier]] = collections.defaultdict(list)
        self._alg_translation: str = ''

    def _replace(self, old: str, new: str, search_from_match: str=None, search_from_match_occurrence: int=None, count: int=1):

        def find_nth(haystack, needle, n):
            start = haystack.lower().find(needle)
            while start >= 0 and n > 1:
                start = haystack.lower().find(needle, start + len(needle))
                n -= 1
            return start
        if search_from_match and search_from_match_occurrence:
            position = find_nth(self._alg_translation, search_from_match, search_from_match_occurrence)
            filedata_pre = self._alg_translation[:position]
            filedata_post = self._alg_translation[position:].replace(old, new, count)
            self._alg_translation = filedata_pre + filedata_post
        else:
            self._alg_translation = self._alg_translation.replace(old, new, count)

    def convert_node_arg(self, node_arg: typing.Union[Identifier, CompValue, Expr, str]) -> str:
        if isinstance(node_arg, Identifier):
            if node_arg in self.aggr_vars.keys():
                grp_var = self.aggr_vars[node_arg].pop(0).n3()
                return grp_var
            else:
                return node_arg.n3()
        elif isinstance(node_arg, CompValue):
            return '{' + node_arg.name + '}'
        elif isinstance(node_arg, Expr):
            return '{' + node_arg.name + '}'
        elif isinstance(node_arg, str):
            return node_arg
        else:
            raise ExpressionNotCoveredException('The expression {0} might not be covered yet.'.format(node_arg))

    def sparql_query_text(self, node):
        """
         https://www.w3.org/TR/sparql11-query/#sparqlSyntax

        :param node:
        :return:
        """
        if isinstance(node, CompValue):
            if node.name == 'SelectQuery':
                self._alg_translation = '-*-SELECT-*- ' + '{' + node.p.name + '}'
            elif node.name == 'BGP':
                triples = ''.join((triple[0].n3() + ' ' + triple[1].n3() + ' ' + triple[2].n3() + '.' for triple in node.triples))
                self._replace('{BGP}', triples)
                self._replace('-*-SELECT-*-', 'SELECT', count=-1)
                self._replace('{GroupBy}', '', count=-1)
                self._replace('{Having}', '', count=-1)
            elif node.name == 'Join':
                self._replace('{Join}', '{' + node.p1.name + '}{' + node.p2.name + '}')
            elif node.name == 'LeftJoin':
                self._replace('{LeftJoin}', '{' + node.p1.name + '}OPTIONAL{{' + node.p2.name + '}}')
            elif node.name == 'Filter':
                if isinstance(node.expr, CompValue):
                    expr = node.expr.name
                else:
                    raise ExpressionNotCoveredException('This expression might not be covered yet.')
                if node.p:
                    if node.p.name == 'AggregateJoin':
                        self._replace('{Filter}', '{' + node.p.name + '}')
                        self._replace('{Having}', 'HAVING({' + expr + '})')
                    else:
                        self._replace('{Filter}', 'FILTER({' + expr + '}) {' + node.p.name + '}')
                else:
                    self._replace('{Filter}', 'FILTER({' + expr + '})')
            elif node.name == 'Union':
                self._replace('{Union}', '{{' + node.p1.name + '}}UNION{{' + node.p2.name + '}}')
            elif node.name == 'Graph':
                expr = 'GRAPH ' + node.term.n3() + ' {{' + node.p.name + '}}'
                self._replace('{Graph}', expr)
            elif node.name == 'Extend':
                query_string = self._alg_translation.lower()
                select_occurrences = query_string.count('-*-select-*-')
                self._replace(node.var.n3(), '(' + self.convert_node_arg(node.expr) + ' as ' + node.var.n3() + ')', search_from_match='-*-select-*-', search_from_match_occurrence=select_occurrences)
                self._replace('{Extend}', '{' + node.p.name + '}')
            elif node.name == 'Minus':
                expr = '{' + node.p1.name + '}MINUS{{' + node.p2.name + '}}'
                self._replace('{Minus}', expr)
            elif node.name == 'Group':
                group_by_vars = []
                if node.expr:
                    for var in node.expr:
                        if isinstance(var, Identifier):
                            group_by_vars.append(var.n3())
                        else:
                            raise ExpressionNotCoveredException('This expression might not be covered yet.')
                    self._replace('{Group}', '{' + node.p.name + '}')
                    self._replace('{GroupBy}', 'GROUP BY ' + ' '.join(group_by_vars) + ' ')
                else:
                    self._replace('{Group}', '{' + node.p.name + '}')
            elif node.name == 'AggregateJoin':
                self._replace('{AggregateJoin}', '{' + node.p.name + '}')
                for agg_func in node.A:
                    if isinstance(agg_func.res, Identifier):
                        identifier = agg_func.res.n3()
                    else:
                        raise ExpressionNotCoveredException('This expression might not be covered yet.')
                    self.aggr_vars[agg_func.res].append(agg_func.vars)
                    agg_func_name = agg_func.name.split('_')[1]
                    distinct = ''
                    if agg_func.distinct:
                        distinct = agg_func.distinct + ' '
                    if agg_func_name == 'GroupConcat':
                        self._replace(identifier, 'GROUP_CONCAT' + '(' + distinct + agg_func.vars.n3() + ';SEPARATOR=' + agg_func.separator.n3() + ')')
                    else:
                        self._replace(identifier, agg_func_name.upper() + '(' + distinct + self.convert_node_arg(agg_func.vars) + ')')
                    self._replace('(SAMPLE({0}) as {0})'.format(self.convert_node_arg(agg_func.vars)), self.convert_node_arg(agg_func.vars))
            elif node.name == 'GroupGraphPatternSub':
                self._replace('GroupGraphPatternSub', ' '.join([self.convert_node_arg(pattern) for pattern in node.part]))
            elif node.name == 'TriplesBlock':
                print('triplesblock')
                self._replace('{TriplesBlock}', ''.join((triple[0].n3() + ' ' + triple[1].n3() + ' ' + triple[2].n3() + '.' for triple in node.triples)))
            elif node.name == 'ToList':
                raise ExpressionNotCoveredException('This expression might not be covered yet.')
            elif node.name == 'OrderBy':
                order_conditions = []
                for c in node.expr:
                    if isinstance(c.expr, Identifier):
                        var = c.expr.n3()
                        if c.order is not None:
                            cond = c.order + '(' + var + ')'
                        else:
                            cond = var
                        order_conditions.append(cond)
                    else:
                        raise ExpressionNotCoveredException('This expression might not be covered yet.')
                self._replace('{OrderBy}', '{' + node.p.name + '}')
                self._replace('{OrderConditions}', ' '.join(order_conditions) + ' ')
            elif node.name == 'Project':
                project_variables = []
                for var in node.PV:
                    if isinstance(var, Identifier):
                        project_variables.append(var.n3())
                    else:
                        raise ExpressionNotCoveredException('This expression might not be covered yet.')
                order_by_pattern = ''
                if node.p.name == 'OrderBy':
                    order_by_pattern = 'ORDER BY {OrderConditions}'
                self._replace('{Project}', ' '.join(project_variables) + '{{' + node.p.name + '}}' + '{GroupBy}' + order_by_pattern + '{Having}')
            elif node.name == 'Distinct':
                self._replace('{Distinct}', 'DISTINCT {' + node.p.name + '}')
            elif node.name == 'Reduced':
                self._replace('{Reduced}', 'REDUCED {' + node.p.name + '}')
            elif node.name == 'Slice':
                slice = 'OFFSET ' + str(node.start) + ' LIMIT ' + str(node.length)
                self._replace('{Slice}', '{' + node.p.name + '}' + slice)
            elif node.name == 'ToMultiSet':
                if node.p.name == 'values':
                    self._replace('{ToMultiSet}', '{{' + node.p.name + '}}')
                else:
                    self._replace('{ToMultiSet}', '{-*-SELECT-*- ' + '{' + node.p.name + '}' + '}')
            elif node.name == 'RelationalExpression':
                expr = self.convert_node_arg(node.expr)
                op = node.op
                if isinstance(list, type(node.other)):
                    other = '(' + ', '.join((self.convert_node_arg(expr) for expr in node.other)) + ')'
                else:
                    other = self.convert_node_arg(node.other)
                condition = '{left} {operator} {right}'.format(left=expr, operator=op, right=other)
                self._replace('{RelationalExpression}', condition)
            elif node.name == 'ConditionalAndExpression':
                inner_nodes = ' && '.join([self.convert_node_arg(expr) for expr in node.other])
                self._replace('{ConditionalAndExpression}', self.convert_node_arg(node.expr) + ' && ' + inner_nodes)
            elif node.name == 'ConditionalOrExpression':
                inner_nodes = ' || '.join([self.convert_node_arg(expr) for expr in node.other])
                self._replace('{ConditionalOrExpression}', '(' + self.convert_node_arg(node.expr) + ' || ' + inner_nodes + ')')
            elif node.name == 'MultiplicativeExpression':
                left_side = self.convert_node_arg(node.expr)
                multiplication = left_side
                for i, operator in enumerate(node.op):
                    multiplication += operator + ' ' + self.convert_node_arg(node.other[i]) + ' '
                self._replace('{MultiplicativeExpression}', multiplication)
            elif node.name == 'AdditiveExpression':
                left_side = self.convert_node_arg(node.expr)
                addition = left_side
                for i, operator in enumerate(node.op):
                    addition += operator + ' ' + self.convert_node_arg(node.other[i]) + ' '
                self._replace('{AdditiveExpression}', addition)
            elif node.name == 'UnaryNot':
                self._replace('{UnaryNot}', '!' + self.convert_node_arg(node.expr))
            elif node.name.endswith('BOUND'):
                bound_var = self.convert_node_arg(node.arg)
                self._replace('{Builtin_BOUND}', 'bound(' + bound_var + ')')
            elif node.name.endswith('IF'):
                arg2 = self.convert_node_arg(node.arg2)
                arg3 = self.convert_node_arg(node.arg3)
                if_expression = 'IF(' + '{' + node.arg1.name + '}, ' + arg2 + ', ' + arg3 + ')'
                self._replace('{Builtin_IF}', if_expression)
            elif node.name.endswith('COALESCE'):
                self._replace('{Builtin_COALESCE}', 'COALESCE(' + ', '.join((self.convert_node_arg(arg) for arg in node.arg)) + ')')
            elif node.name.endswith('Builtin_EXISTS'):
                self._replace('{Builtin_EXISTS}', 'EXISTS ' + '{{' + node.graph.name + '}}')
                traverse(node.graph, visitPre=self.sparql_query_text)
                return node.graph
            elif node.name.endswith('Builtin_NOTEXISTS'):
                print(node.graph.name)
                self._replace('{Builtin_NOTEXISTS}', 'NOT EXISTS ' + '{{' + node.graph.name + '}}')
                traverse(node.graph, visitPre=self.sparql_query_text)
                return node.graph
            elif node.name.endswith('sameTerm'):
                self._replace('{Builtin_sameTerm}', 'SAMETERM(' + self.convert_node_arg(node.arg1) + ', ' + self.convert_node_arg(node.arg2) + ')')
            elif node.name.endswith('Builtin_isIRI'):
                self._replace('{Builtin_isIRI}', 'isIRI(' + self.convert_node_arg(node.arg) + ')')
            elif node.name.endswith('Builtin_isBLANK'):
                self._replace('{Builtin_isBLANK}', 'isBLANK(' + self.convert_node_arg(node.arg) + ')')
            elif node.name.endswith('Builtin_isLITERAL'):
                self._replace('{Builtin_isLITERAL}', 'isLITERAL(' + self.convert_node_arg(node.arg) + ')')
            elif node.name.endswith('Builtin_isNUMERIC'):
                self._replace('{Builtin_isNUMERIC}', 'isNUMERIC(' + self.convert_node_arg(node.arg) + ')')
            elif node.name.endswith('Builtin_STR'):
                self._replace('{Builtin_STR}', 'STR(' + self.convert_node_arg(node.arg) + ')')
            elif node.name.endswith('Builtin_LANG'):
                self._replace('{Builtin_LANG}', 'LANG(' + self.convert_node_arg(node.arg) + ')')
            elif node.name.endswith('Builtin_DATATYPE'):
                self._replace('{Builtin_DATATYPE}', 'DATATYPE(' + self.convert_node_arg(node.arg) + ')')
            elif node.name.endswith('Builtin_IRI'):
                self._replace('{Builtin_IRI}', 'IRI(' + self.convert_node_arg(node.arg) + ')')
            elif node.name.endswith('Builtin_BNODE'):
                self._replace('{Builtin_BNODE}', 'BNODE(' + self.convert_node_arg(node.arg) + ')')
            elif node.name.endswith('STRDT'):
                self._replace('{Builtin_STRDT}', 'STRDT(' + self.convert_node_arg(node.arg1) + ', ' + self.convert_node_arg(node.arg2) + ')')
            elif node.name.endswith('Builtin_STRLANG'):
                self._replace('{Builtin_STRLANG}', 'STRLANG(' + self.convert_node_arg(node.arg1) + ', ' + self.convert_node_arg(node.arg2) + ')')
            elif node.name.endswith('Builtin_UUID'):
                self._replace('{Builtin_UUID}', 'UUID()')
            elif node.name.endswith('Builtin_STRUUID'):
                self._replace('{Builtin_STRUUID}', 'STRUUID()')
            elif node.name.endswith('Builtin_STRLEN'):
                self._replace('{Builtin_STRLEN}', 'STRLEN(' + self.convert_node_arg(node.arg) + ')')
            elif node.name.endswith('Builtin_SUBSTR'):
                args = [self.convert_node_arg(node.arg), node.start]
                if node.length:
                    args.append(node.length)
                expr = 'SUBSTR(' + ', '.join(args) + ')'
                self._replace('{Builtin_SUBSTR}', expr)
            elif node.name.endswith('Builtin_UCASE'):
                self._replace('{Builtin_UCASE}', 'UCASE(' + self.convert_node_arg(node.arg) + ')')
            elif node.name.endswith('Builtin_LCASE'):
                self._replace('{Builtin_LCASE}', 'LCASE(' + self.convert_node_arg(node.arg) + ')')
            elif node.name.endswith('Builtin_STRSTARTS'):
                self._replace('{Builtin_STRSTARTS}', 'STRSTARTS(' + self.convert_node_arg(node.arg1) + ', ' + self.convert_node_arg(node.arg2) + ')')
            elif node.name.endswith('Builtin_STRENDS'):
                self._replace('{Builtin_STRENDS}', 'STRENDS(' + self.convert_node_arg(node.arg1) + ', ' + self.convert_node_arg(node.arg2) + ')')
            elif node.name.endswith('Builtin_CONTAINS'):
                self._replace('{Builtin_CONTAINS}', 'CONTAINS(' + self.convert_node_arg(node.arg1) + ', ' + self.convert_node_arg(node.arg2) + ')')
            elif node.name.endswith('Builtin_STRBEFORE'):
                self._replace('{Builtin_STRBEFORE}', 'STRBEFORE(' + self.convert_node_arg(node.arg1) + ', ' + self.convert_node_arg(node.arg2) + ')')
            elif node.name.endswith('Builtin_STRAFTER'):
                self._replace('{Builtin_STRAFTER}', 'STRAFTER(' + self.convert_node_arg(node.arg1) + ', ' + self.convert_node_arg(node.arg2) + ')')
            elif node.name.endswith('Builtin_ENCODE_FOR_URI'):
                self._replace('{Builtin_ENCODE_FOR_URI}', 'ENCODE_FOR_URI(' + self.convert_node_arg(node.arg) + ')')
            elif node.name.endswith('Builtin_CONCAT'):
                expr = 'CONCAT({vars})'.format(vars=', '.join((self.convert_node_arg(elem) for elem in node.arg)))
                self._replace('{Builtin_CONCAT}', expr)
            elif node.name.endswith('Builtin_LANGMATCHES'):
                self._replace('{Builtin_LANGMATCHES}', 'LANGMATCHES(' + self.convert_node_arg(node.arg1) + ', ' + self.convert_node_arg(node.arg2) + ')')
            elif node.name.endswith('REGEX'):
                args = [self.convert_node_arg(node.text), self.convert_node_arg(node.pattern)]
                expr = 'REGEX(' + ', '.join(args) + ')'
                self._replace('{Builtin_REGEX}', expr)
            elif node.name.endswith('REPLACE'):
                self._replace('{Builtin_REPLACE}', 'REPLACE(' + self.convert_node_arg(node.arg) + ', ' + self.convert_node_arg(node.pattern) + ', ' + self.convert_node_arg(node.replacement) + ')')
            elif node.name == 'Builtin_ABS':
                self._replace('{Builtin_ABS}', 'ABS(' + self.convert_node_arg(node.arg) + ')')
            elif node.name == 'Builtin_ROUND':
                self._replace('{Builtin_ROUND}', 'ROUND(' + self.convert_node_arg(node.arg) + ')')
            elif node.name == 'Builtin_CEIL':
                self._replace('{Builtin_CEIL}', 'CEIL(' + self.convert_node_arg(node.arg) + ')')
            elif node.name == 'Builtin_FLOOR':
                self._replace('{Builtin_FLOOR}', 'FLOOR(' + self.convert_node_arg(node.arg) + ')')
            elif node.name == 'Builtin_RAND':
                self._replace('{Builtin_RAND}', 'RAND()')
            elif node.name == 'Builtin_NOW':
                self._replace('{Builtin_NOW}', 'NOW()')
            elif node.name == 'Builtin_YEAR':
                self._replace('{Builtin_YEAR}', 'YEAR(' + self.convert_node_arg(node.arg) + ')')
            elif node.name == 'Builtin_MONTH':
                self._replace('{Builtin_MONTH}', 'MONTH(' + self.convert_node_arg(node.arg) + ')')
            elif node.name == 'Builtin_DAY':
                self._replace('{Builtin_DAY}', 'DAY(' + self.convert_node_arg(node.arg) + ')')
            elif node.name == 'Builtin_HOURS':
                self._replace('{Builtin_HOURS}', 'HOURS(' + self.convert_node_arg(node.arg) + ')')
            elif node.name == 'Builtin_MINUTES':
                self._replace('{Builtin_MINUTES}', 'MINUTES(' + self.convert_node_arg(node.arg) + ')')
            elif node.name == 'Builtin_SECONDS':
                self._replace('{Builtin_SECONDS}', 'SECONDS(' + self.convert_node_arg(node.arg) + ')')
            elif node.name == 'Builtin_TIMEZONE':
                self._replace('{Builtin_TIMEZONE}', 'TIMEZONE(' + self.convert_node_arg(node.arg) + ')')
            elif node.name == 'Builtin_TZ':
                self._replace('{Builtin_TZ}', 'TZ(' + self.convert_node_arg(node.arg) + ')')
            elif node.name == 'Builtin_MD5':
                self._replace('{Builtin_MD5}', 'MD5(' + self.convert_node_arg(node.arg) + ')')
            elif node.name == 'Builtin_SHA1':
                self._replace('{Builtin_SHA1}', 'SHA1(' + self.convert_node_arg(node.arg) + ')')
            elif node.name == 'Builtin_SHA256':
                self._replace('{Builtin_SHA256}', 'SHA256(' + self.convert_node_arg(node.arg) + ')')
            elif node.name == 'Builtin_SHA384':
                self._replace('{Builtin_SHA384}', 'SHA384(' + self.convert_node_arg(node.arg) + ')')
            elif node.name == 'Builtin_SHA512':
                self._replace('{Builtin_SHA512}', 'SHA512(' + self.convert_node_arg(node.arg) + ')')
            elif node.name == 'values':
                columns = []
                for key in node.res[0].keys():
                    if isinstance(key, Identifier):
                        columns.append(key.n3())
                    else:
                        raise ExpressionNotCoveredException('The expression {0} might not be covered yet.'.format(key))
                values = 'VALUES (' + ' '.join(columns) + ')'
                rows = ''
                for elem in node.res:
                    row = []
                    for term in elem.values():
                        if isinstance(term, Identifier):
                            row.append(term.n3())
                        elif isinstance(term, str):
                            row.append(term)
                        else:
                            raise ExpressionNotCoveredException('The expression {0} might not be covered yet.'.format(term))
                    rows += '(' + ' '.join(row) + ')'
                self._replace('values', values + '{' + rows + '}')
            elif node.name == 'ServiceGraphPattern':
                self._replace('{ServiceGraphPattern}', 'SERVICE ' + self.convert_node_arg(node.term) + '{' + node.graph.name + '}')
                traverse(node.graph, visitPre=self.sparql_query_text)
                return node.graph

    def translateAlgebra(self) -> str:
        traverse(self.query_algebra.algebra, visitPre=self.sparql_query_text)
        return self._alg_translation