from __future__ import annotations
import typing as t
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name, name_sequence
def _explode_to_unnest(expression: exp.Expression) -> exp.Expression:
    if isinstance(expression, exp.Select):
        from sqlglot.optimizer.scope import Scope
        taken_select_names = set(expression.named_selects)
        taken_source_names = {name for name, _ in Scope(expression).references}

        def new_name(names: t.Set[str], name: str) -> str:
            name = find_new_name(names, name)
            names.add(name)
            return name
        arrays: t.List[exp.Condition] = []
        series_alias = new_name(taken_select_names, 'pos')
        series = exp.alias_(exp.Unnest(expressions=[exp.GenerateSeries(start=exp.Literal.number(index_offset))]), new_name(taken_source_names, '_u'), table=[series_alias])
        for select in list(expression.selects):
            explode = select.find(exp.Explode)
            if explode:
                pos_alias = ''
                explode_alias = ''
                if isinstance(select, exp.Alias):
                    explode_alias = select.args['alias']
                    alias = select
                elif isinstance(select, exp.Aliases):
                    pos_alias = select.aliases[0]
                    explode_alias = select.aliases[1]
                    alias = select.replace(exp.alias_(select.this, '', copy=False))
                else:
                    alias = select.replace(exp.alias_(select, ''))
                    explode = alias.find(exp.Explode)
                    assert explode
                is_posexplode = isinstance(explode, exp.Posexplode)
                explode_arg = explode.this
                if isinstance(explode, exp.ExplodeOuter):
                    bracket = explode_arg[0]
                    bracket.set('safe', True)
                    bracket.set('offset', True)
                    explode_arg = exp.func('IF', exp.func('ARRAY_SIZE', exp.func('COALESCE', explode_arg, exp.Array())).eq(0), exp.array(bracket, copy=False), explode_arg)
                if isinstance(explode_arg, exp.Column):
                    taken_select_names.add(explode_arg.output_name)
                unnest_source_alias = new_name(taken_source_names, '_u')
                if not explode_alias:
                    explode_alias = new_name(taken_select_names, 'col')
                    if is_posexplode:
                        pos_alias = new_name(taken_select_names, 'pos')
                if not pos_alias:
                    pos_alias = new_name(taken_select_names, 'pos')
                alias.set('alias', exp.to_identifier(explode_alias))
                series_table_alias = series.args['alias'].this
                column = exp.If(this=exp.column(series_alias, table=series_table_alias).eq(exp.column(pos_alias, table=unnest_source_alias)), true=exp.column(explode_alias, table=unnest_source_alias))
                explode.replace(column)
                if is_posexplode:
                    expressions = expression.expressions
                    expressions.insert(expressions.index(alias) + 1, exp.If(this=exp.column(series_alias, table=series_table_alias).eq(exp.column(pos_alias, table=unnest_source_alias)), true=exp.column(pos_alias, table=unnest_source_alias)).as_(pos_alias))
                    expression.set('expressions', expressions)
                if not arrays:
                    if expression.args.get('from'):
                        expression.join(series, copy=False, join_type='CROSS')
                    else:
                        expression.from_(series, copy=False)
                size: exp.Condition = exp.ArraySize(this=explode_arg.copy())
                arrays.append(size)
                expression.join(exp.alias_(exp.Unnest(expressions=[explode_arg.copy()], offset=exp.to_identifier(pos_alias)), unnest_source_alias, table=[explode_alias]), join_type='CROSS', copy=False)
                if index_offset != 1:
                    size = size - 1
                expression.where(exp.column(series_alias, table=series_table_alias).eq(exp.column(pos_alias, table=unnest_source_alias)).or_((exp.column(series_alias, table=series_table_alias) > size).and_(exp.column(pos_alias, table=unnest_source_alias).eq(size))), copy=False)
        if arrays:
            end: exp.Condition = exp.Greatest(this=arrays[0], expressions=arrays[1:])
            if index_offset != 1:
                end = end - (1 - index_offset)
            series.expressions[0].set('end', end)
    return expression