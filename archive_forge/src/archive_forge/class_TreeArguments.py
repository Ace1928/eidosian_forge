import re
from itertools import zip_longest
from parso.python import tree
from jedi import debug
from jedi.inference.utils import PushBackIterator
from jedi.inference import analysis
from jedi.inference.lazy_value import LazyKnownValue, LazyKnownValues, \
from jedi.inference.names import ParamName, TreeNameDefinition, AnonymousParamName
from jedi.inference.base_value import NO_VALUES, ValueSet, ContextualizedNode
from jedi.inference.value import iterable
from jedi.inference.cache import inference_state_as_method_param_cache
class TreeArguments(AbstractArguments):

    def __init__(self, inference_state, context, argument_node, trailer=None):
        """
        :param argument_node: May be an argument_node or a list of nodes.
        """
        self.argument_node = argument_node
        self.context = context
        self._inference_state = inference_state
        self.trailer = trailer

    @classmethod
    @inference_state_as_method_param_cache()
    def create_cached(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def unpack(self, funcdef=None):
        named_args = []
        for star_count, el in unpack_arglist(self.argument_node):
            if star_count == 1:
                arrays = self.context.infer_node(el)
                iterators = [_iterate_star_args(self.context, a, el, funcdef) for a in arrays]
                for values in list(zip_longest(*iterators)):
                    yield (None, get_merged_lazy_value([v for v in values if v is not None]))
            elif star_count == 2:
                arrays = self.context.infer_node(el)
                for dct in arrays:
                    yield from _star_star_dict(self.context, dct, el, funcdef)
            elif el.type == 'argument':
                c = el.children
                if len(c) == 3:
                    named_args.append((c[0].value, LazyTreeValue(self.context, c[2])))
                else:
                    sync_comp_for = el.children[1]
                    if sync_comp_for.type == 'comp_for':
                        sync_comp_for = sync_comp_for.children[1]
                    comp = iterable.GeneratorComprehension(self._inference_state, defining_context=self.context, sync_comp_for_node=sync_comp_for, entry_node=el.children[0])
                    yield (None, LazyKnownValue(comp))
            else:
                yield (None, LazyTreeValue(self.context, el))
        yield from named_args

    def _as_tree_tuple_objects(self):
        for star_count, argument in unpack_arglist(self.argument_node):
            default = None
            if argument.type == 'argument':
                if len(argument.children) == 3:
                    argument, default = argument.children[::2]
            yield (argument, default, star_count)

    def iter_calling_names_with_star(self):
        for name, default, star_count in self._as_tree_tuple_objects():
            if not star_count or not isinstance(name, tree.Name):
                continue
            yield TreeNameDefinition(self.context, name)

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, self.argument_node)

    def get_calling_nodes(self):
        old_arguments_list = []
        arguments = self
        while arguments not in old_arguments_list:
            if not isinstance(arguments, TreeArguments):
                break
            old_arguments_list.append(arguments)
            for calling_name in reversed(list(arguments.iter_calling_names_with_star())):
                names = calling_name.goto()
                if len(names) != 1:
                    break
                if isinstance(names[0], AnonymousParamName):
                    return []
                if not isinstance(names[0], ParamName):
                    break
                executed_param_name = names[0].get_executed_param_name()
                arguments = executed_param_name.arguments
                break
        if arguments.argument_node is not None:
            return [ContextualizedNode(arguments.context, arguments.argument_node)]
        if arguments.trailer is not None:
            return [ContextualizedNode(arguments.context, arguments.trailer)]
        return []