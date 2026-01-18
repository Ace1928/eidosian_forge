from functools import singledispatch
from outlines.fsm.guide import RegexGuide
from outlines.generate.api import SequenceGenerator
from outlines.integrations.llamacpp import RegexLogitsProcessor
from outlines.models import OpenAI
from outlines.models.llamacpp import LlamaCpp, LlamaSequenceGenerator
from outlines.samplers import Sampler, multinomial
Generate structured text in the language of a regular expression.

    Parameters
    ----------
    model:
        An instance of `Transformer` that represents a model from the
        `transformers` library.
    regex_str:
        The regular expression that the output must follow.
    sampler:
        The sampling algorithm to use to generate token ids from the logits
        distribution.

    Returns
    -------
    A `SequenceGenerator` instance that generates text constrained by the
    regular expression.

    