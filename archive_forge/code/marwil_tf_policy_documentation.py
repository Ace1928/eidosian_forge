import logging
from typing import Any, Dict, List, Optional, Type, Union
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution
from ray.rllib.policy.dynamic_tf_policy_v2 import DynamicTFPolicyV2
from ray.rllib.policy.eager_tf_policy_v2 import EagerTFPolicyV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_mixins import (
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, get_variable
from ray.rllib.utils.tf_utils import explained_variance
from ray.rllib.utils.typing import (
Construct a MARWILTFPolicy inheriting either dynamic or eager base policies.

    Args:
        base: Base class for this policy. DynamicTFPolicyV2 or EagerTFPolicyV2.

    Returns:
        A TF Policy to be used with MAML.
    