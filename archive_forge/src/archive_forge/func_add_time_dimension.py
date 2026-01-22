import logging
import numpy as np
import tree  # pip install dm_tree
from typing import List, Optional
import functools
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.debug import summarize
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.typing import TensorType, ViewRequirementsDict
from ray.util import log_once
from ray.rllib.utils.typing import SampleBatchType
@DeveloperAPI
def add_time_dimension(padded_inputs: TensorType, *, seq_lens: TensorType, framework: str='tf', time_major: bool=False):
    """Adds a time dimension to padded inputs.

    Args:
        padded_inputs: a padded batch of sequences. That is,
            for seq_lens=[1, 2, 2], then inputs=[A, *, B, B, C, C], where
            A, B, C are sequence elements and * denotes padding.
        seq_lens: A 1D tensor of sequence lengths, denoting the non-padded length
            in timesteps of each rollout in the batch.
        framework: The framework string ("tf2", "tf", "torch").
        time_major: Whether data should be returned in time-major (TxB)
            format or not (BxT).

    Returns:
        TensorType: Reshaped tensor of shape [B, T, ...] or [T, B, ...].
    """
    if framework in ['tf2', 'tf']:
        assert time_major is False, 'time-major not supported yet for tf!'
        padded_inputs = tf.convert_to_tensor(padded_inputs)
        padded_batch_size = tf.shape(padded_inputs)[0]
        new_batch_size = tf.shape(seq_lens)[0]
        time_size = padded_batch_size // new_batch_size
        new_shape = tf.concat([tf.expand_dims(new_batch_size, axis=0), tf.expand_dims(time_size, axis=0), tf.shape(padded_inputs)[1:]], axis=0)
        return tf.reshape(padded_inputs, new_shape)
    elif framework == 'torch':
        padded_inputs = torch.as_tensor(padded_inputs)
        padded_batch_size = padded_inputs.shape[0]
        new_batch_size = seq_lens.shape[0]
        time_size = padded_batch_size // new_batch_size
        batch_major_shape = (new_batch_size, time_size) + padded_inputs.shape[1:]
        padded_outputs = padded_inputs.view(batch_major_shape)
        if time_major:
            padded_outputs = padded_outputs.transpose(0, 1)
        return padded_outputs
    else:
        assert framework == 'np', 'Unknown framework: {}'.format(framework)
        padded_inputs = np.asarray(padded_inputs)
        padded_batch_size = padded_inputs.shape[0]
        new_batch_size = seq_lens.shape[0]
        time_size = padded_batch_size // new_batch_size
        batch_major_shape = (new_batch_size, time_size) + padded_inputs.shape[1:]
        padded_outputs = padded_inputs.reshape(batch_major_shape)
        if time_major:
            padded_outputs = padded_outputs.transpose(0, 1)
        return padded_outputs