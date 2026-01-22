import inspect
from collections.abc import Iterable
from typing import Optional, Text
def _evaluate_qnode(self, x):
    """Evaluates a QNode for a single input datapoint.

        Args:
            x (tensor): the datapoint

        Returns:
            tensor: output datapoint
        """
    kwargs = {**{self.input_arg: x}, **{k: 1.0 * w for k, w in self.qnode_weights.items()}}
    res = self.qnode(**kwargs)
    if isinstance(res, (list, tuple)):
        if len(x.shape) > 1:
            res = [tf.reshape(r, (tf.shape(x)[0], tf.reduce_prod(r.shape[1:]))) for r in res]
        return tf.experimental.numpy.hstack(res)
    return res