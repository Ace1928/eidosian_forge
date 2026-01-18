import logging
from typing import Dict, List, Tuple, Type, Union
from ray.rllib.algorithms.simple_q.utils import make_q_models
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import Categorical, TFActionDistribution
from ray.rllib.policy.dynamic_tf_policy_v2 import DynamicTFPolicyV2
from ray.rllib.policy.eager_tf_policy_v2 import EagerTFPolicyV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_mixins import (
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.tf_utils import huber_loss
from ray.rllib.utils.typing import LocalOptimizer, ModelGradients, TensorType
Returns the learning rate in a stats dict.

            Args:
                policy: The Policy object.
                train_batch: The data used for training.

            Returns:
                Dict[str, TensorType]: The stats dict.
            