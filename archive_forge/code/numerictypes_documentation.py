import numbers
import warnings
from .multiarray import (
from .._utils import set_module
from ._string_helpers import (
from ._type_aliases import (
from ._dtype import _kind_name
from builtins import bool, int, float, complex, object, str, bytes
from numpy.compat import long, unicode

    Determine common type following standard coercion rules.

    .. deprecated:: NumPy 1.25

        This function is deprecated, use `numpy.promote_types` or
        `numpy.result_type` instead.  To achieve semantics for the
        `scalar_types` argument, use `numpy.result_type` and pass the Python
        values `0`, `0.0`, or `0j`.
        This will give the same results in almost all cases.
        More information and rare exception can be found in the
        `NumPy 1.25 release notes <https://numpy.org/devdocs/release/1.25.0-notes.html>`_.

    Parameters
    ----------
    array_types : sequence
        A list of dtypes or dtype convertible objects representing arrays.
    scalar_types : sequence
        A list of dtypes or dtype convertible objects representing scalars.

    Returns
    -------
    datatype : dtype
        The common data type, which is the maximum of `array_types` ignoring
        `scalar_types`, unless the maximum of `scalar_types` is of a
        different kind (`dtype.kind`). If the kind is not understood, then
        None is returned.

    See Also
    --------
    dtype, common_type, can_cast, mintypecode

    Examples
    --------
    >>> np.find_common_type([], [np.int64, np.float32, complex])
    dtype('complex128')
    >>> np.find_common_type([np.int64, np.float32], [])
    dtype('float64')

    The standard casting rules ensure that a scalar cannot up-cast an
    array unless the scalar is of a fundamentally different kind of data
    (i.e. under a different hierarchy in the data type hierarchy) then
    the array:

    >>> np.find_common_type([np.float32], [np.int64, np.float64])
    dtype('float32')

    Complex is of a different type, so it up-casts the float in the
    `array_types` argument:

    >>> np.find_common_type([np.float32], [complex])
    dtype('complex128')

    Type specifier strings are convertible to dtypes and can therefore
    be used instead of dtypes:

    >>> np.find_common_type(['f4', 'f4', 'i4'], ['c8'])
    dtype('complex128')

    