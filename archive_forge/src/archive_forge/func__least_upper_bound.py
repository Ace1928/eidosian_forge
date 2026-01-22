import functools
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend.common.variables import ALLOWED_DTYPES
from keras.src.backend.common.variables import standardize_dtype
@functools.lru_cache(512)
def _least_upper_bound(*nodes):
    """Compute the least upper bound of a set of nodes.

    Args:
        nodes: sequence of entries from dtypes + weak_types

    Returns:
        The type representing the least upper bound of the input nodes on the
        promotion lattice.
    """
    N = set(nodes)
    UB = LATTICE_UPPER_BOUNDS
    try:
        bounds = [UB[n] for n in N]
    except KeyError:
        dtype = next((n for n in N if n not in UB))
        raise ValueError(f'dtype={dtype!r} is not a valid dtype for Keras type promotion.')
    CUB = set.intersection(*bounds)
    LUB = CUB & N or {c for c in CUB if CUB.issubset(UB[c])}
    if len(LUB) == 1:
        return LUB.pop()
    elif len(LUB) == 0:
        msg = f'Input dtypes {tuple((str(n) for n in nodes))} have no available implicit dtype promotion path. Try explicitly casting inputs to the desired output type.'
        raise ValueError(msg)
    else:
        raise ValueError(f"Internal Type Promotion error: {nodes} do not have a unique least upper bound on the specified lattice; options are {LUB}. This is an unexpected error in Keras's internal logic; please report it to the maintainers.")