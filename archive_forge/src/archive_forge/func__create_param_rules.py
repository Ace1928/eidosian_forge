import warnings
def _create_param_rules(cls, func):
    """ Create ply.yacc rules based on a parameterized rule function

    Generates new methods (one per each pair of parameters) based on the
    template rule function `func`, and attaches them to `cls`. The rule
    function's parameters must be accessible via its `_params` attribute.
    """
    for xxx, yyy in func._params:

        def param_rule(self, p):
            func(self, p)
        param_rule.__doc__ = func.__doc__.replace('xxx', xxx).replace('yyy', yyy)
        param_rule.__name__ = func.__name__.replace('xxx', xxx)
        setattr(cls, param_rule.__name__, param_rule)