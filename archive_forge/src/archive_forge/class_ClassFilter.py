from jedi import debug
from jedi.parser_utils import get_cached_parent_scope, expr_is_dotted, \
from jedi.inference.cache import inference_state_method_cache, CachedMetaClass, \
from jedi.inference import compiled
from jedi.inference.lazy_value import LazyKnownValues, LazyTreeValue
from jedi.inference.filters import ParserTreeFilter
from jedi.inference.names import TreeNameDefinition, ValueName
from jedi.inference.arguments import unpack_arglist, ValuesArguments
from jedi.inference.base_value import ValueSet, iterator_to_value_set, \
from jedi.inference.context import ClassContext
from jedi.inference.value.function import FunctionAndClassBase
from jedi.inference.gradual.generics import LazyGenericManager, TupleGenericManager
from jedi.plugins import plugin_manager
class ClassFilter(ParserTreeFilter):

    def __init__(self, class_value, node_context=None, until_position=None, origin_scope=None, is_instance=False):
        super().__init__(class_value.as_context(), node_context, until_position=until_position, origin_scope=origin_scope)
        self._class_value = class_value
        self._is_instance = is_instance

    def _convert_names(self, names):
        return [ClassName(class_value=self._class_value, tree_name=name, name_context=self._node_context, apply_decorators=not self._is_instance) for name in names]

    def _equals_origin_scope(self):
        node = self._origin_scope
        while node is not None:
            if node == self._parser_scope or node == self.parent_context:
                return True
            node = get_cached_parent_scope(self._parso_cache_node, node)
        return False

    def _access_possible(self, name):
        return not name.value.startswith('__') or name.value.endswith('__') or self._equals_origin_scope()

    def _filter(self, names):
        names = super()._filter(names)
        return [name for name in names if self._access_possible(name)]