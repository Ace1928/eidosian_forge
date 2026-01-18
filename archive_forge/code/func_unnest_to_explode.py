from __future__ import annotations
import typing as t
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name, name_sequence
def unnest_to_explode(expression: exp.Expression) -> exp.Expression:
    """Convert cross join unnest into lateral view explode."""
    if isinstance(expression, exp.Select):
        for join in expression.args.get('joins') or []:
            unnest = join.this
            if isinstance(unnest, exp.Unnest):
                alias = unnest.args.get('alias')
                udtf = exp.Posexplode if unnest.args.get('offset') else exp.Explode
                expression.args['joins'].remove(join)
                for e, column in zip(unnest.expressions, alias.columns if alias else []):
                    expression.append('laterals', exp.Lateral(this=udtf(this=e), view=True, alias=exp.TableAlias(this=alias.this, columns=[column])))
    return expression