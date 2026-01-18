import logging
import psutil
from typing import Optional, Any
import numpy as np
from ray.rllib.utils import deprecation_warning
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.deprecation import DEPRECATED_VALUE
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.replay_buffers import (
from ray.rllib.policy.sample_batch import concat_samples, MultiAgentBatch, SampleBatch
from ray.rllib.utils.typing import ResultDict, SampleBatchType, AlgorithmConfigDict
from ray.util import log_once
@DeveloperAPI
def validate_buffer_config(config: dict) -> None:
    """Checks and fixes values in the replay buffer config.

    Checks the replay buffer config for common misconfigurations, warns or raises
    error in case validation fails. The type "key" is changed into the inferred
    replay buffer class.

    Args:
        config: The replay buffer config to be validated.

    Raises:
        ValueError: When detecting severe misconfiguration.
    """
    if config.get('replay_buffer_config', None) is None:
        config['replay_buffer_config'] = {}
    if config.get('worker_side_prioritization', DEPRECATED_VALUE) != DEPRECATED_VALUE:
        deprecation_warning(old="config['worker_side_prioritization']", new="config['replay_buffer_config']['worker_side_prioritization']", error=True)
    prioritized_replay = config.get('prioritized_replay', DEPRECATED_VALUE)
    if prioritized_replay != DEPRECATED_VALUE:
        deprecation_warning(old="config['prioritized_replay'] or config['replay_buffer_config']['prioritized_replay']", help="Replay prioritization specified by config key. RLlib's new replay buffer API requires setting `config['replay_buffer_config']['type']`, e.g. `config['replay_buffer_config']['type'] = 'MultiAgentPrioritizedReplayBuffer'` to change the default behaviour.", error=True)
    capacity = config.get('buffer_size', DEPRECATED_VALUE)
    if capacity == DEPRECATED_VALUE:
        capacity = config['replay_buffer_config'].get('buffer_size', DEPRECATED_VALUE)
    if capacity != DEPRECATED_VALUE:
        deprecation_warning(old="config['buffer_size'] or config['replay_buffer_config']['buffer_size']", new="config['replay_buffer_config']['capacity']", error=True)
    replay_burn_in = config.get('burn_in', DEPRECATED_VALUE)
    if replay_burn_in != DEPRECATED_VALUE:
        config['replay_buffer_config']['replay_burn_in'] = replay_burn_in
        deprecation_warning(old="config['burn_in']", help="config['replay_buffer_config']['replay_burn_in']")
    replay_batch_size = config.get('replay_batch_size', DEPRECATED_VALUE)
    if replay_batch_size == DEPRECATED_VALUE:
        replay_batch_size = config['replay_buffer_config'].get('replay_batch_size', DEPRECATED_VALUE)
    if replay_batch_size != DEPRECATED_VALUE:
        deprecation_warning(old="config['replay_batch_size'] or config['replay_buffer_config']['replay_batch_size']", help='Specification of replay_batch_size is not supported anymore but is derived from `train_batch_size`. Specify the number of items you want to replay upon calling the sample() method of replay buffers if this does not work for you.', error=True)
    keys_with_deprecated_positions = ['prioritized_replay_alpha', 'prioritized_replay_beta', 'prioritized_replay_eps', 'no_local_replay_buffer', 'replay_zero_init_states', 'replay_buffer_shards_colocated_with_driver']
    for k in keys_with_deprecated_positions:
        if config.get(k, DEPRECATED_VALUE) != DEPRECATED_VALUE:
            deprecation_warning(old="config['{}']".format(k), help="config['replay_buffer_config']['{}']".format(k), error=False)
            if config.get('replay_buffer_config') is not None:
                config['replay_buffer_config'][k] = config[k]
    learning_starts = config.get('learning_starts', config.get('replay_buffer_config', {}).get('learning_starts', DEPRECATED_VALUE))
    if learning_starts != DEPRECATED_VALUE:
        deprecation_warning(old="config['learning_starts'] orconfig['replay_buffer_config']['learning_starts']", help="config['num_steps_sampled_before_learning_starts']", error=True)
        config['num_steps_sampled_before_learning_starts'] = learning_starts
    replay_sequence_length = config.get('replay_sequence_length', None)
    if replay_sequence_length is not None:
        config['replay_buffer_config']['replay_sequence_length'] = replay_sequence_length
        deprecation_warning(old="config['replay_sequence_length']", help="Replay sequence length specified at new location config['replay_buffer_config']['replay_sequence_length'] will be overwritten.", error=True)
    replay_buffer_config = config['replay_buffer_config']
    assert 'type' in replay_buffer_config, "Can not instantiate ReplayBuffer from config without 'type' key."
    buffer_type = config['replay_buffer_config']['type']
    if isinstance(buffer_type, str) and buffer_type.find('.') == -1:
        config['replay_buffer_config']['type'] = 'ray.rllib.utils.replay_buffers.' + buffer_type
    dummy_buffer = from_config(buffer_type, config['replay_buffer_config'])
    config['replay_buffer_config']['type'] = type(dummy_buffer)
    if hasattr(dummy_buffer, 'update_priorities'):
        if config['replay_buffer_config'].get('replay_mode', 'independent') == 'lockstep':
            raise ValueError('Prioritized replay is not supported when replay_mode=lockstep.')
        elif config['replay_buffer_config'].get('replay_sequence_length', 0) > 1:
            raise ValueError('Prioritized replay is not supported when replay_sequence_length > 1.')
    elif config['replay_buffer_config'].get('worker_side_prioritization'):
        raise ValueError('Worker side prioritization is not supported when prioritized_replay=False.')