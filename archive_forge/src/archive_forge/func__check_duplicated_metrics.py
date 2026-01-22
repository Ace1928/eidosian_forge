import copy
import tensorflow.compat.v2 as tf
from keras.src import losses as losses_mod
from keras.src import metrics as metrics_mod
from keras.src.saving import saving_lib
from keras.src.utils import generic_utils
from keras.src.utils import losses_utils
from keras.src.utils import tf_utils
def _check_duplicated_metrics(self, metrics, weighted_metrics):
    """Raise error when user provided metrics have any duplications.

        Note that metrics are stateful container, a shared metric instance
        between model.metric and model.weighted_metric will make the same
        intance to be udpated twice, and report wrong value.

        Args:
          metrics: User provided metrics list.
          weighted_metrics: User provided weighted metrics list.

        Raises:
          ValueError, when duplicated metrics instance discovered in user
            provided metrics and weighted metrics.
        """
    seen = set()
    duplicated = []
    for x in tf.nest.flatten(metrics) + tf.nest.flatten(weighted_metrics):
        if not isinstance(x, metrics_mod.Metric):
            continue
        if x in seen:
            duplicated.append(x)
        seen.add(x)
    if duplicated:
        raise ValueError(f'Found duplicated metrics object in the user provided metrics and weighted metrics. This will cause the same metric object to be updated multiple times, and report wrong results. \nDuplicated items: {duplicated}')