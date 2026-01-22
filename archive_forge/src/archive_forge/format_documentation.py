from functools import singledispatch
from outlines.fsm.types import python_types_to_regex
from outlines.generate.api import SequenceGenerator
from outlines.models import OpenAI
from outlines.samplers import Sampler, multinomial
from .regex import regex
Generate structured data that can be parsed as a Python type.

    Parameters
    ----------
    model:
        An instance of `Transformer` that represents a model from the
        `transformers` library.
    python_type:
        A Python type. The output of the generator must be parseable into
        this type.
    sampler:
        The sampling algorithm to use to generate token ids from the logits
        distribution.

    Returns
    -------
    A `SequenceGenerator` instance that generates text constrained by the Python type
    and translates this text into the corresponding type.

    