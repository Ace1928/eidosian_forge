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
@publish_method('items')
def _imitate_items(self, arguments):
    lazy_values = [LazyKnownValue(FakeTuple(self.inference_state, (LazyTreeValue(self._defining_context, key_node), LazyTreeValue(self._defining_context, value_node)))) for key_node, value_node in self.get_tree_entries()]
    return ValueSet([FakeList(self.inference_state, lazy_values)])