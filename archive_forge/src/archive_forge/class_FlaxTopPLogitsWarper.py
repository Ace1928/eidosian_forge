import inspect
import jax
import jax.lax as lax
import jax.numpy as jnp
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
class FlaxTopPLogitsWarper(FlaxLogitsWarper):
    """
    [`FlaxLogitsWarper`] that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <= prob_cut_off.

    Args:
        top_p (`float`):
            If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
            higher are kept for generation.
        filter_value (`float`, *optional*, defaults to -inf):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_p: float, filter_value: float=-float('Inf'), min_tokens_to_keep: int=1):
        if not isinstance(top_p, float) or (top_p < 0 or top_p > 1.0):
            raise ValueError(f'`top_p` has to be a float > 0 and < 1, but is {top_p}')
        if not isinstance(min_tokens_to_keep, int) or min_tokens_to_keep < 1:
            raise ValueError(f'`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}')
        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        topk_scores, topk_indices = lax.top_k(scores, scores.shape[-1])
        mask_scores = jnp.full_like(scores, self.filter_value)
        cumulative_probs = jax.nn.softmax(topk_scores, axis=-1).cumsum(axis=-1)
        score_mask = cumulative_probs < self.top_p
        score_mask = jnp.roll(score_mask, 1)
        score_mask |= score_mask.at[:, 0].set(True)
        score_mask = score_mask.at[:, :self.min_tokens_to_keep].set(True)
        topk_next_scores = jnp.where(score_mask, topk_scores, mask_scores)
        next_scores = jax.lax.sort_key_val(topk_indices, topk_next_scores)[-1]
        return next_scores