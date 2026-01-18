import difflib
import inspect
import pickle
import traceback
from collections import defaultdict
from contextlib import contextmanager
import numpy as np
import param
from .accessors import Opts  # noqa (clean up in 2.0)
from .pprint import InfoPrinter
from .tree import AttrTree
from .util import group_sanitizer, label_sanitizer, sanitize_identifier
@classmethod
def update_backends(cls, id_mapping, custom_trees, backend=None):
    """
        Given the id_mapping from previous ids to new ids and the new
        custom tree dictionary, update the current backend with the
        supplied trees and update the keys in the remaining backends to
        stay linked with the current object.
        """
    backend = Store.current_backend if backend is None else backend
    Store.custom_options(backend=backend).update(custom_trees)
    Store._lookup_cache[backend] = {}
    for b in Store.loaded_backends():
        if b == backend:
            continue
        backend_trees = Store._custom_options[b]
        for old_id, new_id in id_mapping:
            tree = backend_trees.get(old_id, None)
            if tree is not None:
                backend_trees[new_id] = tree