import collections
from ray.rllib.models.tf.tf_action_dist import Categorical
from ray.rllib.utils.framework import try_import_tf
With the selected log_probs for multi-discrete actions of behaviour
    and target policies we compute the log_rhos for calculating the vtrace.