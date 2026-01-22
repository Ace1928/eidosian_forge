import sys
import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export
def _get_seed(_):
    """Wraps TF get_seed to make deterministic random generation easier.

            This makes a variable's initialization (and calls that involve
            random number generation) depend only on how many random number
            generations were used in the scope so far, rather than on how many
            unrelated operations the graph contains.

            Returns:
              Random seed tuple.
            """
    op_seed = self._operation_seed
    if self._mode == 'constant':
        tf.random.set_seed(op_seed)
    else:
        if op_seed in self._observed_seeds:
            raise ValueError('This `DeterministicRandomTestTool` object is trying to re-use the ' + f'already-used operation seed {op_seed}. ' + 'It cannot guarantee random numbers will match ' + 'between eager and sessions when an operation seed ' + 'is reused. You most likely set ' + '`operation_seed` explicitly but used a value that ' + 'caused the naturally-incrementing operation seed ' + 'sequences to overlap with an already-used seed.')
        self._observed_seeds.add(op_seed)
        self._operation_seed += 1
    return (self._seed, op_seed)