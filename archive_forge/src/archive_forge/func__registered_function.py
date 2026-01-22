import itertools
from tensorflow.python.framework import ops
from tensorflow.python.util import tf_inspect
def _registered_function(type_list, registry):
    """Given a list of classes, finds the most specific function registered."""
    enumerated_hierarchies = [enumerate(tf_inspect.getmro(t)) for t in type_list]
    cls_combinations = list(itertools.product(*enumerated_hierarchies))

    def hierarchy_distance(cls_combination):
        candidate_distance = sum((c[0] for c in cls_combination))
        if tuple((c[1] for c in cls_combination)) in registry:
            return candidate_distance
        return 10000
    registered_combination = min(cls_combinations, key=hierarchy_distance)
    return registry.get(tuple((r[1] for r in registered_combination)), None)