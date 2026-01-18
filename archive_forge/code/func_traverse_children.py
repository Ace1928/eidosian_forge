import collections
import collections.abc
import types
import optree
from keras.src.api_export import keras_export
from keras.src.backend.config import backend
def traverse_children():
    children, treedef = optree.tree_flatten(structure, is_leaf=lambda x: x is not structure, none_is_leaf=True, namespace='keras')
    if treedef.num_nodes == 1 and treedef.num_leaves == 1:
        return structure
    else:
        return optree.tree_unflatten(treedef, [traverse(func, c, top_down=top_down) for c in children])