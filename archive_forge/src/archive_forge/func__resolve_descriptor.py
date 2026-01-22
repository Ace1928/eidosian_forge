import types
from pure_eval.utils import of_type, CannotEval
def _resolve_descriptor(d, instance, owner):
    try:
        return type(of_type(d, *safe_descriptor_types)).__get__(d, instance, owner)
    except AttributeError as e:
        raise CannotEval from e