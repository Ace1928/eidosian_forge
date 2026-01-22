import inspect
import jax
import jax.lax as lax
import jax.numpy as jnp
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
class FlaxForcedBOSTokenLogitsProcessor(FlaxLogitsProcessor):
    """
    [`FlaxLogitsProcessor`] that enforces the specified token as the first generated token.

    Args:
        bos_token_id (`int`):
            The id of the token to force as the first generated token.
    """

    def __init__(self, bos_token_id: int):
        self.bos_token_id = bos_token_id

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        new_scores = jnp.full(scores.shape, -float('inf'))
        apply_penalty = 1 - jnp.bool_(cur_len - 1)
        scores = jnp.where(apply_penalty, new_scores.at[:, self.bos_token_id].set(0), scores)
        return scores