import numpy as np
from ray.rllib.algorithms.dreamerv3.utils.debugging import (
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.tf_utils import inverse_symlog
def _report_rewards(*, results, computed_rewards, sampled_rewards, descr_prefix=None, descr_reward):
    descr_prefix = descr_prefix + '_' if descr_prefix else ''
    mse_sampled_vs_computed_rewards = np.mean(np.square(computed_rewards - sampled_rewards))
    mse_sampled_vs_computed_rewards = np.mean(mse_sampled_vs_computed_rewards)
    results.update({f'{descr_prefix}sampled_vs_{descr_reward}_rewards_mse': mse_sampled_vs_computed_rewards})