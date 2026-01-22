from jedi import debug
from jedi.cache import memoize_method
from jedi.inference.utils import to_tuple
from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.inference.value.iterable import SequenceLiteralValue
from jedi.inference.helpers import is_string
def _resolve_forward_references(context, value_set):
    for value in value_set:
        if is_string(value):
            from jedi.inference.gradual.annotation import _get_forward_reference_node
            node = _get_forward_reference_node(context, value.get_safe_value())
            if node is not None:
                for c in context.infer_node(node):
                    yield c
        else:
            yield value