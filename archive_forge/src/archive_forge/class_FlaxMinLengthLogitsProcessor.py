import inspect
import jax
import jax.lax as lax
import jax.numpy as jnp
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
class FlaxMinLengthLogitsProcessor(FlaxLogitsProcessor):
    """
    [`FlaxLogitsProcessor`] enforcing a min-length by setting EOS probability to 0.

    Args:
        min_length (`int`):
            The minimum length below which the score of `eos_token_id` is set to `-float("Inf")`.
        eos_token_id (`int`):
            The id of the *end-of-sequence* token.
    """

    def __init__(self, min_length: int, eos_token_id: int):
        if not isinstance(min_length, int) or min_length < 0:
            raise ValueError(f'`min_length` has to be a positive integer, but is {min_length}')
        if not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise ValueError(f'`eos_token_id` has to be a positive integer, but is {eos_token_id}')
        self.min_length = min_length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        apply_penalty = 1 - jnp.clip(cur_len - self.min_length, 0, 1)
        scores = jnp.where(apply_penalty, scores.at[:, self.eos_token_id].set(-float('inf')), scores)
        return scores