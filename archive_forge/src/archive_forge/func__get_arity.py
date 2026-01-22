import re
def _get_arity(f):
    return sum((1 for p in signature(f).parameters.values() if p.default is Parameter.empty and p.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)))