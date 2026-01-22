import collections
import collections.abc
import types
import optree
from keras.src.api_export import keras_export
from keras.src.backend.config import backend
def _map_structure_with_path_up_to(shallow_structure, func, *structures):
    results = []
    for path_and_values in _multiyield_flat_up_to(shallow_structure, *structures):
        results.append(func(*path_and_values))
    shallow_structure_spec = optree.tree_structure(shallow_structure, none_is_leaf=True, namespace='keras')
    return shallow_structure_spec.unflatten(results)