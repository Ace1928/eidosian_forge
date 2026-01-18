from typing import TYPE_CHECKING, List, Optional, Tuple, Union
import torch
from outlines.models.tokenizer import Tokenizer
def transformers(model_name: str, device: Optional[str]=None, model_kwargs: dict={}, tokenizer_kwargs: dict={}):
    """Instantiate a model from the `transformers` library and its tokenizer.

    Parameters
    ----------
    model_name
        The name of the model as listed on Hugging Face's model page.
    device
        The device(s) on which the model should be loaded. This overrides
        the `device_map` entry in `model_kwargs` when provided.
    model_kwargs
        A dictionary that contains the keyword arguments to pass to the
        `from_pretrained` method when loading the model.
    tokenizer_kwargs
        A dictionary that contains the keyword arguments to pass to the
        `from_pretrained` method when loading the tokenizer.

    Returns
    -------
    A `TransformersModel` model instance.

    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError('The `transformers` library needs to be installed in order to use `transformers` models.')
    if device is not None:
        model_kwargs['device_map'] = device
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer_kwargs.setdefault('padding_side', 'left')
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    return Transformers(model, tokenizer)