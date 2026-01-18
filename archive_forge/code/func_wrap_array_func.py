from __future__ import division
from builtins import next
from builtins import zip
from builtins import range
import sys
import inspect
import numpy
from numpy.core import numeric
import uncertainties.umath_core as umath_core
import uncertainties.core as uncert_core
from uncertainties.core import deprecation
def wrap_array_func(func):
    """
    Return a version of the function func() that works even when
    func() is given a NumPy array that contains numbers with
    uncertainties, as first argument.

    This wrapper is similar to uncertainties.core.wrap(), except that
    it handles an array argument instead of float arguments, and that
    the result can be an array.

    However, the returned function is more restricted: the array
    argument cannot be given as a keyword argument with the name in
    the original function (it is not a drop-in replacement).

    func -- function whose first argument is a single NumPy array,
    and which returns a NumPy array.
    """

    @uncert_core.set_doc('    Version of %s(...) that works even when its first argument is a NumPy\n    array that contains numbers with uncertainties.\n\n    Warning: elements of the first argument array that are not\n    AffineScalarFunc objects must not depend on uncert_core.Variable\n    objects in any way.  Otherwise, the dependence of the result in\n    uncert_core.Variable objects will be incorrect.\n\n    Original documentation:\n    %s' % (func.__name__, func.__doc__))
    def wrapped_func(arr, *args, **kwargs):
        arr_nominal_value = nominal_values(arr)
        func_nominal_value = func(arr_nominal_value, *args, **kwargs)
        variables = set()
        for element in arr.flat:
            if isinstance(element, uncert_core.AffineScalarFunc):
                variables |= set(element.derivatives.keys())
        if not variables:
            return func_nominal_value
        derivatives = numpy.vectorize(lambda _: {})(func_nominal_value)
        for var in variables:
            shift_var = max(var._std_dev / 100000.0, 1e-08 * abs(var._nominal_value))
            if not shift_var:
                shift_var = 1e-08
            shift_arr = array_derivative(arr, var) * shift_var
            shifted_arr_values = arr_nominal_value + shift_arr
            func_shifted = func(shifted_arr_values, *args, **kwargs)
            numerical_deriv = (func_shifted - func_nominal_value) / shift_var
            for derivative_dict, derivative_value in zip(derivatives.flat, numerical_deriv.flat):
                if derivative_value:
                    derivative_dict[var] = derivative_value
        return numpy.vectorize(uncert_core.AffineScalarFunc)(func_nominal_value, numpy.vectorize(uncert_core.LinearCombination)(derivatives))
    wrapped_func = uncert_core.set_doc('    Version of %s(...) that works even when its first argument is a NumPy\n    array that contains numbers with uncertainties.\n\n    Warning: elements of the first argument array that are not\n    AffineScalarFunc objects must not depend on uncert_core.Variable\n    objects in any way.  Otherwise, the dependence of the result in\n    uncert_core.Variable objects will be incorrect.\n\n    Original documentation:\n    %s' % (func.__name__, func.__doc__))(wrapped_func)
    wrapped_func.__name__ = func.__name__
    return wrapped_func