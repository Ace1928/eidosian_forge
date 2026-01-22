from collections import OrderedDict, deque
import numpy as np
from ray.rllib.utils import force_list
from ray.rllib.utils.framework import try_import_tf
def _assign_weights(self, weights):
    """Sets weigths using exact or closest assignable variable name

        Args:
            weights: Dictionary mapping variable names to their
                weights.
        Returns:
            Tuple[List, Dict]: assigned variables list, dict of
                placeholders and weights
        """
    assigned = []
    feed_dict = {}
    assignable = set(self.assignment_nodes.keys())

    def nb_common_elem(l1, l2):
        return len([e for e in l1 if e in l2])

    def assign(name, value):
        feed_dict[self.placeholders[name]] = value
        assigned.append(name)
        assignable.remove(name)
    for name, value in weights.items():
        if name in assignable:
            assign(name, value)
        else:
            common = {var: nb_common_elem(name.split('/'), var.split('/')) for var in assignable}
            select = [close_var for close_var, cn in sorted(common.items(), key=lambda i: -i[1]) if cn > 0 and value.shape == self.assignment_nodes[close_var].shape]
            if select:
                assign(select[0], value)
    assert assigned, 'No variables in the input matched those in the network. Possible cause: Two networks were defined in the same TensorFlow graph. To fix this, place each network definition in its own tf.Graph.'
    assert len(assigned) == len(weights), "All weights couldn't be assigned because no variable had an exact/close name or had same shape"
    return ([self.assignment_nodes[v] for v in assigned], feed_dict)