import re
import warnings
from parso import parse, ParserSyntaxError
from jedi import debug
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.base_value import iterator_to_value_set, ValueSet, \
from jedi.inference.lazy_value import LazyKnownValues
def _expand_typestr(type_str):
    """
    Attempts to interpret the possible types in `type_str`
    """
    if re.search('\\bor\\b', type_str):
        for t in type_str.split('or'):
            yield t.split('of')[0].strip()
    elif re.search('\\bof\\b', type_str):
        yield type_str.split('of')[0]
    elif type_str.startswith('{'):
        node = parse(type_str, version='3.7').children[0]
        if node.type == 'atom':
            for leaf in getattr(node.children[1], 'children', []):
                if leaf.type == 'number':
                    if '.' in leaf.value:
                        yield 'float'
                    else:
                        yield 'int'
                elif leaf.type == 'string':
                    if 'b' in leaf.string_prefix.lower():
                        yield 'bytes'
                    else:
                        yield 'str'
    else:
        yield type_str