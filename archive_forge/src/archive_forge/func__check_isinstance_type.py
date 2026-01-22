from parso.tree import search_ancestor
from parso.python.tree import Name
from jedi import settings
from jedi.inference.arguments import TreeArguments
from jedi.inference.value import iterable
from jedi.inference.base_value import NO_VALUES
from jedi.parser_utils import is_scope
def _check_isinstance_type(value, node, search_name):
    lazy_cls = None
    trailer = _get_isinstance_trailer_arglist(node)
    if trailer is not None and len(trailer.children) == 3:
        arglist = trailer.children[1]
        args = TreeArguments(value.inference_state, value, arglist, trailer)
        param_list = list(args.unpack())
        if len(param_list) == 2 and len(arglist.children) == 3:
            (key1, _), (key2, lazy_value_cls) = param_list
            if key1 is None and key2 is None:
                call = _get_call_string(search_name)
                is_instance_call = _get_call_string(arglist.children[0])
                if call == is_instance_call:
                    lazy_cls = lazy_value_cls
    if lazy_cls is None:
        return None
    value_set = NO_VALUES
    for cls_or_tup in lazy_cls.infer():
        if isinstance(cls_or_tup, iterable.Sequence) and cls_or_tup.array_type == 'tuple':
            for lazy_value in cls_or_tup.py__iter__():
                value_set |= lazy_value.infer().execute_with_values()
        else:
            value_set |= cls_or_tup.execute_with_values()
    return value_set