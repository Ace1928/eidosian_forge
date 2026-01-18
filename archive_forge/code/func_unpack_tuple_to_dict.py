from jedi.inference import compiled
from jedi.inference import analysis
from jedi.inference.lazy_value import LazyKnownValue, LazyKnownValues, \
from jedi.inference.helpers import get_int_or_none, is_string, \
from jedi.inference.utils import safe_property, to_list
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.filters import LazyAttributeOverwrite, publish_method
from jedi.inference.base_value import ValueSet, Value, NO_VALUES, \
from jedi.parser_utils import get_sync_comp_fors
from jedi.inference.context import CompForContext
from jedi.inference.value.dynamic_arrays import check_array_additions
def unpack_tuple_to_dict(context, types, exprlist):
    """
    Unpacking tuple assignments in for statements and expr_stmts.
    """
    if exprlist.type == 'name':
        return {exprlist.value: types}
    elif exprlist.type == 'atom' and exprlist.children[0] in ('(', '['):
        return unpack_tuple_to_dict(context, types, exprlist.children[1])
    elif exprlist.type in ('testlist', 'testlist_comp', 'exprlist', 'testlist_star_expr'):
        dct = {}
        parts = iter(exprlist.children[::2])
        n = 0
        for lazy_value in types.iterate(ContextualizedNode(context, exprlist)):
            n += 1
            try:
                part = next(parts)
            except StopIteration:
                analysis.add(context, 'value-error-too-many-values', part, message='ValueError: too many values to unpack (expected %s)' % n)
            else:
                dct.update(unpack_tuple_to_dict(context, lazy_value.infer(), part))
        has_parts = next(parts, None)
        if types and has_parts is not None:
            analysis.add(context, 'value-error-too-few-values', has_parts, message='ValueError: need more than %s values to unpack' % n)
        return dct
    elif exprlist.type == 'power' or exprlist.type == 'atom_expr':
        return {}
    elif exprlist.type == 'star_expr':
        return {}
    raise NotImplementedError