from __future__ import annotations
from typing import Any
from sympy.external import import_module
from sympy.printing.printer import Printer
from sympy.utilities.iterables import is_sequence
import sympy
from functools import partial
def aesara_function(inputs, outputs, scalar=False, *, dim=None, dims=None, broadcastables=None, **kwargs):
    """
    Create a Aesara function from SymPy expressions.

    The inputs and outputs are converted to Aesara variables using
    :func:`.aesara_code` and then passed to ``aesara.function``.

    Parameters
    ==========

    inputs
        Sequence of symbols which constitute the inputs of the function.

    outputs
        Sequence of expressions which constitute the outputs(s) of the
        function. The free symbols of each expression must be a subset of
        ``inputs``.

    scalar : bool
        Convert 0-dimensional arrays in output to scalars. This will return a
        Python wrapper function around the Aesara function object.

    cache : dict
        Cached Aesara variables (see :class:`AesaraPrinter.cache
        <AesaraPrinter>`). Defaults to the module-level global cache.

    dtypes : dict
        Passed to :meth:`.AesaraPrinter.doprint`.

    broadcastables : dict
        Passed to :meth:`.AesaraPrinter.doprint`.

    dims : dict
        Alternative to ``broadcastables`` argument. Mapping from elements of
        ``inputs`` to integers indicating the dimension of their associated
        arrays/tensors. Overrides ``broadcastables`` argument if given.

    dim : int
        Another alternative to the ``broadcastables`` argument. Common number of
        dimensions to use for all arrays/tensors.
        ``aesara_function([x, y], [...], dim=2)`` is equivalent to using
        ``broadcastables={x: (False, False), y: (False, False)}``.

    Returns
    =======
    callable
        A callable object which takes values of ``inputs`` as positional
        arguments and returns an output array for each of the expressions
        in ``outputs``. If ``outputs`` is a single expression the function will
        return a Numpy array, if it is a list of multiple expressions the
        function will return a list of arrays. See description of the ``squeeze``
        argument above for the behavior when a single output is passed in a list.
        The returned object will either be an instance of
        ``aesara.compile.function.types.Function`` or a Python wrapper
        function around one. In both cases, the returned value will have a
        ``aesara_function`` attribute which points to the return value of
        ``aesara.function``.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.printing.aesaracode import aesara_function

    A simple function with one input and one output:

    >>> f1 = aesara_function([x], [x**2 - 1], scalar=True)
    >>> f1(3)
    8.0

    A function with multiple inputs and one output:

    >>> f2 = aesara_function([x, y, z], [(x**z + y**z)**(1/z)], scalar=True)
    >>> f2(3, 4, 2)
    5.0

    A function with multiple inputs and multiple outputs:

    >>> f3 = aesara_function([x, y], [x**2 + y**2, x**2 - y**2], scalar=True)
    >>> f3(2, 3)
    [13.0, -5.0]

    See also
    ========

    dim_handling

    """
    if not aesara:
        raise ImportError('Aesara is required for aesara_function')
    cache = kwargs.pop('cache', {})
    dtypes = kwargs.pop('dtypes', {})
    broadcastables = dim_handling(inputs, dim=dim, dims=dims, broadcastables=broadcastables)
    code = partial(aesara_code, cache=cache, dtypes=dtypes, broadcastables=broadcastables)
    tinputs = list(map(code, inputs))
    toutputs = list(map(code, outputs))
    toutputs = [output if isinstance(output, aesara.graph.basic.Variable) else aet.as_tensor_variable(output) for output in toutputs]
    if len(toutputs) == 1:
        toutputs = toutputs[0]
    func = aesara.function(tinputs, toutputs, **kwargs)
    is_0d = [len(o.variable.broadcastable) == 0 for o in func.outputs]
    if not scalar or not any(is_0d):
        func.aesara_function = func
        return func

    def wrapper(*args):
        out = func(*args)
        if is_sequence(out):
            return [o[()] if is_0d[i] else o for i, o in enumerate(out)]
        else:
            return out[()]
    wrapper.__wrapped__ = func
    wrapper.__doc__ = func.__doc__
    wrapper.aesara_function = func
    return wrapper