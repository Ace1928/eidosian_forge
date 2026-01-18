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
def timeslice_along_seq_lens_with_overlap(sample_batch: SampleBatchType, seq_lens: Optional[List[int]]=None, zero_pad_max_seq_len: int=0, pre_overlap: int=0, zero_init_states: bool=True) -> List['SampleBatch']:
    """Slices batch along `seq_lens` (each seq-len item produces one batch).

    Args:
        sample_batch: The SampleBatch to timeslice.
        seq_lens (Optional[List[int]]): An optional list of seq_lens to slice
            at. If None, use `sample_batch[SampleBatch.SEQ_LENS]`.
        zero_pad_max_seq_len: If >0, already zero-pad the resulting
            slices up to this length. NOTE: This max-len will include the
            additional timesteps gained via setting pre_overlap (see Example).
        pre_overlap: If >0, will overlap each two consecutive slices by
            this many timesteps (toward the left side). This will cause
            zero-padding at the very beginning of the batch.
        zero_init_states: Whether initial states should always be
            zero'd. If False, will use the state_outs of the batch to
            populate state_in values.

    Returns:
        List[SampleBatch]: The list of (new) SampleBatches.

    Examples:
        assert seq_lens == [5, 5, 2]
        assert sample_batch.count == 12
        # self = 0 1 2 3 4 | 5 6 7 8 9 | 10 11 <- timesteps
        slices = timeslice_along_seq_lens_with_overlap(
            sample_batch=sample_batch.
            zero_pad_max_seq_len=10,
            pre_overlap=3)
        # Z = zero padding (at beginning or end).
        #             |pre (3)|     seq     | max-seq-len (up to 10)
        # slices[0] = | Z Z Z |  0  1 2 3 4 | Z Z
        # slices[1] = | 2 3 4 |  5  6 7 8 9 | Z Z
        # slices[2] = | 7 8 9 | 10 11 Z Z Z | Z Z
        # Note that `zero_pad_max_seq_len=10` includes the 3 pre-overlaps
        #  count (makes sure each slice has exactly length 10).
    """
    if seq_lens is None:
        seq_lens = sample_batch.get(SampleBatch.SEQ_LENS)
    elif sample_batch.get(SampleBatch.SEQ_LENS) is not None and log_once('overriding_sequencing_information'):
        logger.warning('Found sequencing information in a batch that will be ignored when slicing. Ignore this warning if you know what you are doing.')
    if seq_lens is None:
        max_seq_len = zero_pad_max_seq_len - pre_overlap
        if log_once('no_sequence_lengths_available_for_time_slicing'):
            logger.warning('Trying to slice a batch along sequences without sequence lengths being provided in the batch. Batch will be sliced into slices of size {} = {} - {} = zero_pad_max_seq_len - pre_overlap.'.format(max_seq_len, zero_pad_max_seq_len, pre_overlap))
        num_seq_lens, last_seq_len = divmod(len(sample_batch), max_seq_len)
        seq_lens = [zero_pad_max_seq_len] * num_seq_lens + ([last_seq_len] if last_seq_len else [])
    assert seq_lens is not None and len(seq_lens) > 0, 'Cannot timeslice along `seq_lens` when `seq_lens` is empty or None!'
    start = 0
    slices = []
    for seq_len in seq_lens:
        pre_begin = start - pre_overlap
        slice_begin = start
        end = start + seq_len
        slices.append((pre_begin, slice_begin, end))
        start += seq_len
    timeslices = []
    for begin, slice_begin, end in slices:
        zero_length = None
        data_begin = 0
        zero_init_states_ = zero_init_states
        if begin < 0:
            zero_length = pre_overlap
            data_begin = slice_begin
            zero_init_states_ = True
        else:
            eps_ids = sample_batch[SampleBatch.EPS_ID][begin if begin >= 0 else 0:end]
            is_last_episode_ids = eps_ids == eps_ids[-1]
            if not is_last_episode_ids[0]:
                zero_length = int(sum(1.0 - is_last_episode_ids))
                data_begin = begin + zero_length
                zero_init_states_ = True
        if zero_length is not None:
            data = {k: np.concatenate([np.zeros(shape=(zero_length,) + v.shape[1:], dtype=v.dtype), v[data_begin:end]]) for k, v in sample_batch.items() if k != SampleBatch.SEQ_LENS}
        else:
            data = {k: v[begin:end] for k, v in sample_batch.items() if k != SampleBatch.SEQ_LENS}
        if zero_init_states_:
            i = 0
            key = 'state_in_{}'.format(i)
            while key in data:
                data[key] = np.zeros_like(sample_batch[key][0:1])
                data.pop('state_out_{}'.format(i), None)
                i += 1
                key = 'state_in_{}'.format(i)
        else:
            i = 0
            key = 'state_in_{}'.format(i)
            while key in data:
                data[key] = sample_batch['state_out_{}'.format(i)][begin - 1:begin]
                del data['state_out_{}'.format(i)]
                i += 1
                key = 'state_in_{}'.format(i)
        timeslices.append(SampleBatch(data, seq_lens=[end - begin]))
    if zero_pad_max_seq_len > 0:
        for ts in timeslices:
            ts.right_zero_pad(max_seq_len=zero_pad_max_seq_len, exclude_states=True)
    return timeslices