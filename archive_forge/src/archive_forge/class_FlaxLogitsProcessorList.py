import inspect
import jax
import jax.lax as lax
import jax.numpy as jnp
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
class FlaxLogitsProcessorList(list):
    """
    This class can be used to create a list of [`FlaxLogitsProcessor`] or [`FlaxLogitsWarper`] to subsequently process
    a `scores` input tensor. This class inherits from list and adds a specific *__call__* method to apply each
    [`FlaxLogitsProcessor`] or [`FlaxLogitsWarper`] to the inputs.
    """

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int, **kwargs) -> jnp.ndarray:
        for processor in self:
            function_args = inspect.signature(processor.__call__).parameters
            if len(function_args) > 3:
                if not all((arg in kwargs for arg in list(function_args.keys())[2:])):
                    raise ValueError(f'Make sure that all the required parameters: {list(function_args.keys())} for {processor.__class__} are passed to the logits processor.')
                scores = processor(input_ids, scores, cur_len, **kwargs)
            else:
                scores = processor(input_ids, scores, cur_len)
        return scores