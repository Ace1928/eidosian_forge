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
def unpack_arglist(arglist):
    if arglist is None:
        return
    if arglist.type != 'arglist' and (not (arglist.type == 'argument' and arglist.children[0] in ('*', '**'))):
        yield (0, arglist)
        return
    iterator = iter(arglist.children)
    for child in iterator:
        if child == ',':
            continue
        elif child in ('*', '**'):
            c = next(iterator, None)
            assert c is not None
            yield (len(child.value), c)
        elif child.type == 'argument' and child.children[0] in ('*', '**'):
            assert len(child.children) == 2
            yield (len(child.children[0].value), child.children[1])
        else:
            yield (0, child)