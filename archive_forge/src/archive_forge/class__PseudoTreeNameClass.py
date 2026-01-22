from jedi.inference.cache import inference_state_method_cache
from jedi.inference.base_value import ValueSet, NO_VALUES, Value, \
from jedi.inference.compiled import builtin_from_name
from jedi.inference.value.klass import ClassFilter
from jedi.inference.value.klass import ClassMixin
from jedi.inference.utils import to_list
from jedi.inference.names import AbstractNameDefinition, ValueName
from jedi.inference.context import ClassContext
from jedi.inference.gradual.generics import TupleGenericManager
class _PseudoTreeNameClass(Value):
    """
    In typeshed, some classes are defined like this:

        Tuple: _SpecialForm = ...

    Now this is not a real class, therefore we have to do some workarounds like
    this class. Essentially this class makes it possible to goto that `Tuple`
    name, without affecting anything else negatively.
    """
    api_type = 'class'

    def __init__(self, parent_context, tree_name):
        super().__init__(parent_context.inference_state, parent_context)
        self._tree_name = tree_name

    @property
    def tree_node(self):
        return self._tree_name

    def get_filters(self, *args, **kwargs):

        class EmptyFilter(ClassFilter):

            def __init__(self):
                pass

            def get(self, name, **kwargs):
                return []

            def values(self, **kwargs):
                return []
        yield EmptyFilter()

    def py__class__(self):
        return builtin_from_name(self.inference_state, 'type')

    @property
    def name(self):
        return ValueName(self, self._tree_name)

    def get_qualified_names(self):
        return (self._tree_name.value,)

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self._tree_name.value)