from functools import wraps
from jedi import debug
def _memoize_default(default=_NO_DEFAULT, inference_state_is_first_arg=False, second_arg_is_inference_state=False):
    """ This is a typical memoization decorator, BUT there is one difference:
    To prevent recursion it sets defaults.

    Preventing recursion is in this case the much bigger use than speed. I
    don't think, that there is a big speed difference, but there are many cases
    where recursion could happen (think about a = b; b = a).
    """

    def func(function):

        def wrapper(obj, *args, **kwargs):
            if inference_state_is_first_arg:
                cache = obj.memoize_cache
            elif second_arg_is_inference_state:
                cache = args[0].memoize_cache
            else:
                cache = obj.inference_state.memoize_cache
            try:
                memo = cache[function]
            except KeyError:
                cache[function] = memo = {}
            key = (obj, args, frozenset(kwargs.items()))
            if key in memo:
                return memo[key]
            else:
                if default is not _NO_DEFAULT:
                    memo[key] = default
                rv = function(obj, *args, **kwargs)
                memo[key] = rv
                return rv
        return wrapper
    return func