import functools
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend.common.variables import ALLOWED_DTYPES
from keras.src.backend.common.variables import standardize_dtype
def _make_lattice_upper_bounds():
    lattice = _type_promotion_lattice()
    upper_bounds = {node: {node} for node in lattice}
    for n in lattice:
        while True:
            new_upper_bounds = set().union(*(lattice[b] for b in upper_bounds[n]))
            if n in new_upper_bounds:
                raise ValueError(f'cycle detected in type promotion lattice for node {n}')
            if new_upper_bounds.issubset(upper_bounds[n]):
                break
            upper_bounds[n] |= new_upper_bounds
    return upper_bounds