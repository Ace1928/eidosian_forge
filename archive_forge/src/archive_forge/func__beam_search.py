import copy
import inspect
import warnings
from functools import partial
from typing import Any, Dict, Optional, Union
import flax
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from ..models.auto import (
from ..utils import ModelOutput, logging
from .configuration_utils import GenerationConfig
from .flax_logits_process import (
def _beam_search(self, input_ids: None, max_length: Optional[int]=None, pad_token_id: Optional[int]=None, eos_token_id: Optional[int]=None, length_penalty: Optional[float]=None, early_stopping: Optional[Union[bool, str]]=None, logits_processor: Optional[FlaxLogitsProcessorList]=None, trace: bool=True, params: Optional[Dict[str, jnp.ndarray]]=None, num_return_sequences: Optional[int]=None, model_kwargs: Optional[Dict[str, jnp.ndarray]]=None):
    """
        This beam search function is heavily inspired by Flax's official example:
        https://github.com/google/flax/blob/main/examples/wmt/decode.py
        """

    def flatten_beam_dim(tensor):
        """Flattens the first two dimensions of a non-scalar array."""
        if tensor.ndim == 0:
            return tensor
        return tensor.reshape((tensor.shape[0] * tensor.shape[1],) + tensor.shape[2:])

    def unflatten_beam_dim(tensor, batch_size, num_beams):
        """Unflattens the first, flat batch*beam dimension of a non-scalar array."""
        if tensor.ndim == 0:
            return tensor
        return tensor.reshape((batch_size, num_beams) + tensor.shape[1:])

    def gather_beams(nested, beam_indices, batch_size, new_num_beams):
        """
            Gathers the beam slices indexed by beam_indices into new beam array.
            """
        batch_indices = jnp.reshape(jnp.arange(batch_size * new_num_beams) // new_num_beams, (batch_size, new_num_beams))

        def gather_fn(tensor):
            if tensor.ndim == 0:
                return tensor
            else:
                return tensor[batch_indices, beam_indices]
        return jax.tree_util.tree_map(gather_fn, nested)
    max_length = max_length if max_length is not None else self.generation_config.max_length
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    length_penalty = length_penalty if length_penalty is not None else self.generation_config.length_penalty
    early_stopping = early_stopping if early_stopping is not None else self.generation_config.early_stopping
    num_return_sequences = num_return_sequences if num_return_sequences is not None else self.generation_config.num_return_sequences
    batch_size, num_beams, cur_len = input_ids.shape
    eos_token_id = jnp.array(eos_token_id, dtype=jnp.int32 if eos_token_id is not None else None)
    pad_token_id = jnp.array(pad_token_id, dtype=jnp.int32)
    cur_len = jnp.array(cur_len)
    decoder_prompt_len = input_ids.shape[-1]
    sequences = jnp.full((batch_size, num_beams, max_length), pad_token_id, dtype=jnp.int32)
    running_sequences = jnp.full((batch_size, num_beams, max_length), pad_token_id, dtype=jnp.int32)
    running_sequences = lax.dynamic_update_slice(sequences, input_ids, (0, 0, 0))
    is_sent_finished = jnp.zeros((batch_size, num_beams), dtype=jnp.bool_)
    running_scores = jnp.tile(jnp.array([0.0] + [np.array(-10000000.0)] * (num_beams - 1)), [batch_size, 1])
    scores = jnp.ones((batch_size, num_beams)) * np.array(-10000000.0)
    model = self.decode if self.config.is_encoder_decoder else self
    if 'encoder_outputs' in model_kwargs:
        model_kwargs['encoder_outputs']['last_hidden_state'] = flatten_beam_dim(model_kwargs['encoder_outputs']['last_hidden_state'])
    for kwarg in ['attention_mask', 'decoder_attention_mask']:
        if kwarg in model_kwargs:
            model_kwargs[kwarg] = flatten_beam_dim(model_kwargs[kwarg])
    model_kwargs = self.prepare_inputs_for_generation(flatten_beam_dim(input_ids), max_length, **model_kwargs)
    state = BeamSearchState(cur_len=cur_len, running_sequences=running_sequences, running_scores=running_scores, sequences=sequences, scores=scores, is_sent_finished=is_sent_finished, model_kwargs=model_kwargs)

    def beam_search_cond_fn(state):
        """beam search state termination condition fn."""
        not_max_length_yet = state.cur_len < max_length
        if early_stopping == 'never' and length_penalty > 0.0:
            best_running_score = state.running_scores[:, :1] / (max_length - decoder_prompt_len) ** length_penalty
        else:
            best_running_score = state.running_scores[:, :1] / (state.cur_len - decoder_prompt_len) ** length_penalty
        worst_finished_score = jnp.where(state.is_sent_finished, jnp.min(state.scores, axis=1, keepdims=True), np.array(-10000000.0))
        improvement_still_possible = jnp.any(best_running_score > worst_finished_score)
        still_open_beam = ~(jnp.all(state.is_sent_finished) & (early_stopping is True))
        return not_max_length_yet & still_open_beam & improvement_still_possible

    def beam_search_body_fn(state, input_ids_length=1):
        """beam search state update fn."""
        input_token = flatten_beam_dim(lax.dynamic_slice(state.running_sequences, (0, 0, state.cur_len - input_ids_length), (batch_size, num_beams, input_ids_length)))
        model_outputs = model(input_token, params=params, **state.model_kwargs)
        logits = unflatten_beam_dim(model_outputs.logits[:, -1], batch_size, num_beams)
        cache = jax.tree_util.tree_map(lambda tensor: unflatten_beam_dim(tensor, batch_size, num_beams), model_outputs.past_key_values)
        logits = self._adapt_logits_for_beam_search(logits)
        log_probs = jax.nn.log_softmax(logits)
        log_probs = logits_processor(flatten_beam_dim(running_sequences), flatten_beam_dim(log_probs), state.cur_len)
        log_probs = unflatten_beam_dim(log_probs, batch_size, num_beams)
        log_probs = log_probs + jnp.expand_dims(state.running_scores, axis=2)
        vocab_size = log_probs.shape[2]
        log_probs = log_probs.reshape((batch_size, num_beams * vocab_size))
        beams_to_keep = 2 * num_beams
        topk_log_probs, topk_indices = lax.top_k(log_probs, k=beams_to_keep)
        topk_beam_indices = topk_indices // vocab_size
        topk_running_sequences = gather_beams(state.running_sequences, topk_beam_indices, batch_size, beams_to_keep)
        topk_ids = jnp.expand_dims(topk_indices % vocab_size, axis=2)
        topk_sequences = lax.dynamic_update_slice(topk_running_sequences, topk_ids, (0, 0, state.cur_len))
        did_topk_just_finished = topk_sequences[:, :, state.cur_len] == eos_token_id
        running_topk_log_probs = topk_log_probs + did_topk_just_finished * np.array(-10000000.0)
        next_topk_indices = lax.top_k(running_topk_log_probs, k=num_beams)[1]
        next_running_sequences, next_running_scores = gather_beams([topk_sequences, running_topk_log_probs], next_topk_indices, batch_size, num_beams)
        topk_log_probs = topk_log_probs / (state.cur_len + 1 - decoder_prompt_len) ** length_penalty
        beams_in_batch_are_full = jnp.broadcast_to(state.is_sent_finished.all(axis=-1, keepdims=True), did_topk_just_finished.shape) & (early_stopping is True)
        add_penalty = ~did_topk_just_finished | beams_in_batch_are_full
        topk_log_probs += add_penalty * np.array(-10000000.0)
        merged_sequences = jnp.concatenate([state.sequences, topk_sequences], axis=1)
        merged_scores = jnp.concatenate([state.scores, topk_log_probs], axis=1)
        merged_is_sent_finished = jnp.concatenate([state.is_sent_finished, did_topk_just_finished], axis=1)
        topk_merged_indices = lax.top_k(merged_scores, k=num_beams)[1]
        next_sequences, next_scores, next_is_sent_finished = gather_beams([merged_sequences, merged_scores, merged_is_sent_finished], topk_merged_indices, batch_size, num_beams)
        next_running_indices = gather_beams(topk_beam_indices, next_topk_indices, batch_size, num_beams)
        next_cache = gather_beams(cache, next_running_indices, batch_size, num_beams)
        model_outputs['past_key_values'] = jax.tree_util.tree_map(lambda x: flatten_beam_dim(x), next_cache)
        next_model_kwargs = self.update_inputs_for_generation(model_outputs, state.model_kwargs)
        return BeamSearchState(cur_len=state.cur_len + 1, running_scores=next_running_scores, running_sequences=next_running_sequences, scores=next_scores, sequences=next_sequences, is_sent_finished=next_is_sent_finished, model_kwargs=next_model_kwargs)
    state = partial(beam_search_body_fn, input_ids_length=input_ids.shape[-1])(state)
    if not trace:
        state = self._run_loop_in_debug(beam_search_cond_fn, beam_search_body_fn, state)
    else:
        state = lax.while_loop(beam_search_cond_fn, beam_search_body_fn, state)
    none_finished = jnp.any(state.is_sent_finished, axis=1)
    sequences = jnp.where(none_finished[:, None, None], state.sequences, state.running_sequences)
    scores = jnp.where(none_finished[:, None], state.scores, state.running_scores)
    sequences = flatten_beam_dim(sequences[:, :num_return_sequences, :])
    scores = flatten_beam_dim(scores[:, :num_return_sequences])
    return FlaxBeamSearchOutput(sequences=sequences, scores=scores)