from functools import reduce
from operator import add
from itertools import zip_longest
from parso.python.tree import Name
from jedi import debug
from jedi.parser_utils import clean_scope_docstring
from jedi.inference.helpers import SimpleGetItemNotFound
from jedi.inference.utils import safe_property
from jedi.inference.cache import inference_state_as_method_param_cache
from jedi.cache import memoize_method
class HelperValueMixin:

    def get_root_context(self):
        value = self
        if value.parent_context is None:
            return value.as_context()
        while True:
            if value.parent_context is None:
                return value
            value = value.parent_context

    def execute(self, arguments):
        return self.inference_state.execute(self, arguments=arguments)

    def execute_with_values(self, *value_list):
        from jedi.inference.arguments import ValuesArguments
        arguments = ValuesArguments([ValueSet([value]) for value in value_list])
        return self.inference_state.execute(self, arguments)

    def execute_annotation(self):
        return self.execute_with_values()

    def gather_annotation_classes(self):
        return ValueSet([self])

    def merge_types_of_iterate(self, contextualized_node=None, is_async=False):
        return ValueSet.from_sets((lazy_value.infer() for lazy_value in self.iterate(contextualized_node, is_async)))

    def _get_value_filters(self, name_or_str):
        origin_scope = name_or_str if isinstance(name_or_str, Name) else None
        yield from self.get_filters(origin_scope=origin_scope)
        if self.is_stub():
            from jedi.inference.gradual.conversion import convert_values
            for c in convert_values(ValueSet({self})):
                yield from c.get_filters()

    def goto(self, name_or_str, name_context=None, analysis_errors=True):
        from jedi.inference import finder
        filters = self._get_value_filters(name_or_str)
        names = finder.filter_name(filters, name_or_str)
        debug.dbg('context.goto %s in (%s): %s', name_or_str, self, names)
        return names

    def py__getattribute__(self, name_or_str, name_context=None, position=None, analysis_errors=True):
        """
        :param position: Position of the last statement -> tuple of line, column
        """
        if name_context is None:
            name_context = self
        names = self.goto(name_or_str, name_context, analysis_errors)
        values = ValueSet.from_sets((name.infer() for name in names))
        if not values:
            n = name_or_str.value if isinstance(name_or_str, Name) else name_or_str
            values = self.py__getattribute__alternatives(n)
        if not names and (not values) and analysis_errors:
            if isinstance(name_or_str, Name):
                from jedi.inference import analysis
                analysis.add_attribute_error(name_context, self, name_or_str)
        debug.dbg('context.names_to_types: %s -> %s', names, values)
        return values

    def py__await__(self):
        await_value_set = self.py__getattribute__('__await__')
        if not await_value_set:
            debug.warning('Tried to run __await__ on value %s', self)
        return await_value_set.execute_with_values()

    def py__name__(self):
        return self.name.string_name

    def iterate(self, contextualized_node=None, is_async=False):
        debug.dbg('iterate %s', self)
        if is_async:
            from jedi.inference.lazy_value import LazyKnownValues
            return iter([LazyKnownValues(self.py__getattribute__('__aiter__').execute_with_values().py__getattribute__('__anext__').execute_with_values().py__getattribute__('__await__').execute_with_values().py__stop_iteration_returns())])
        return self.py__iter__(contextualized_node)

    def is_sub_class_of(self, class_value):
        with debug.increase_indent_cm('subclass matching of %s <=> %s' % (self, class_value), color='BLUE'):
            for cls in self.py__mro__():
                if cls.is_same_class(class_value):
                    debug.dbg('matched subclass True', color='BLUE')
                    return True
            debug.dbg('matched subclass False', color='BLUE')
            return False

    def is_same_class(self, class2):
        if type(class2).is_same_class != HelperValueMixin.is_same_class:
            return class2.is_same_class(self)
        return self == class2

    @memoize_method
    def as_context(self, *args, **kwargs):
        return self._as_context(*args, **kwargs)