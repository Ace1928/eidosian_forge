from jedi import debug
from jedi import settings
from jedi.inference import recursion
from jedi.inference.base_value import ValueSet, NO_VALUES, HelperValueMixin, \
from jedi.inference.lazy_value import LazyKnownValues
from jedi.inference.helpers import infer_call_of_leaf
from jedi.inference.cache import inference_state_method_cache
class _DynamicArrayAdditions(HelperValueMixin):
    """
    Used for the usage of set() and list().
    This is definitely a hack, but a good one :-)
    It makes it possible to use set/list conversions.

    This is not a proper context, because it doesn't have to be. It's not used
    in the wild, it's just used within typeshed as an argument to `__init__`
    for set/list and never used in any other place.
    """

    def __init__(self, instance, arguments):
        self._instance = instance
        self._arguments = arguments

    def py__class__(self):
        tuple_, = self._instance.inference_state.builtins_module.py__getattribute__('tuple')
        return tuple_

    def py__iter__(self, contextualized_node=None):
        arguments = self._arguments
        try:
            _, lazy_value = next(arguments.unpack())
        except StopIteration:
            pass
        else:
            yield from lazy_value.infer().iterate()
        from jedi.inference.arguments import TreeArguments
        if isinstance(arguments, TreeArguments):
            additions = _internal_check_array_additions(arguments.context, self._instance)
            yield from additions

    def iterate(self, contextualized_node=None, is_async=False):
        return self.py__iter__(contextualized_node)