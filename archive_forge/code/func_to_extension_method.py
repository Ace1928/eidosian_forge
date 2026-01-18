import collections
import re
import sys
from yaql.language import exceptions
from yaql.language import lexer
def to_extension_method(name, context):
    layers = context.collect_functions(name, lambda t, ctx: not t.is_function or not t.is_method, use_convention=True)
    if len(layers) > 1:
        raise ValueError('Multi layer functions are not supported by this helper method')
    if len(layers) > 0:
        for spec in layers[0]:
            spec = spec.clone()
            spec.is_function = True
            spec.is_method = True
            yield spec