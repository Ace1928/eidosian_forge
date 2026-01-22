from warnings import warn
class Annotate:
    """
    Construct syntax highlighted annotation for a given jitted function:

    Example:

    >>> import numba
    >>> from numba.pretty_annotate import Annotate
    >>> @numba.jit
    ... def test(q):
    ...     res = 0
    ...     for i in range(q):
    ...         res += i
    ...     return res
    ...
    >>> test(10)
    45
    >>> Annotate(test)

    The last line will return an HTML and/or ANSI representation that will be
    displayed accordingly in Jupyter/IPython.

    Function annotations persist across compilation for newly encountered
    type signatures and as a result annotations are shown for all signatures
    by default.

    Annotations for a specific signature can be shown by using the
    ``signature`` parameter.

    >>> @numba.jit
    ... def add(x, y):
    ...     return x + y
    ...
    >>> add(1, 2)
    3
    >>> add(1.3, 5.7)
    7.0
    >>> add.signatures
    [(int64, int64), (float64, float64)]
    >>> Annotate(add, signature=add.signatures[1])  # annotation for (float64, float64)
    """

    def __init__(self, function, signature=None, **kwargs):
        style = kwargs.get('style', 'default')
        if not function.signatures:
            raise ValueError('function need to be jitted for at least one signature')
        ann = function.get_annotation_info(signature=signature)
        self.ann = ann
        for k, v in ann.items():
            res = hllines(reform_code(v), style)
            rest = htlines(reform_code(v), style)
            v['pygments_lines'] = [(a, b, c, d) for (a, b), c, d in zip(v['python_lines'], res, rest)]

    def _repr_html_(self):
        return get_html_template().render(func_data=self.ann)

    def __repr__(self):
        return get_ansi_template().render(func_data=self.ann)