from typing import TYPE_CHECKING, Dict, List, Literal, Tuple, Union
import torch
from torch import Tensor
from torchmetrics.functional.multimodal.clip_score import _get_clip_model_and_processor
from torchmetrics.utilities.checks import _SKIP_SLOW_DOCTEST, _try_proceed_with_timeout
from torchmetrics.utilities.imports import _PIQ_GREATER_EQUAL_0_8, _TRANSFORMERS_GREATER_EQUAL_4_10
def _clip_iqa_format_prompts(prompts: Tuple[Union[str, Tuple[str, str]]]=('quality',)) -> Tuple[List[str], List[str]]:
    """Converts the provided keywords into a list of prompts for the model to calculate the anchor vectors.

    Args:
        prompts: A string, tuple of strings or nested tuple of strings. If a single string is provided, it must be one
            of the available prompts (see above). Else the input is expected to be a tuple, where each element can
            be one of two things: either a string or a tuple of strings. If a string is provided, it must be one of the
            available prompts (see above). If tuple is provided, it must be of length 2 and the first string must be a
            positive prompt and the second string must be a negative prompt.

    Returns:
        Tuple containing a list of prompts and a list of the names of the prompts. The first list is double the length
        of the second list.

    Examples::

        >>> # single prompt
        >>> _clip_iqa_format_prompts(("quality",))
        (['Good photo.', 'Bad photo.'], ['quality'])
        >>> # multiple prompts
        >>> _clip_iqa_format_prompts(("quality", "brightness"))
        (['Good photo.', 'Bad photo.', 'Bright photo.', 'Dark photo.'], ['quality', 'brightness'])
        >>> # Custom prompts
        >>> _clip_iqa_format_prompts(("quality", ("Super good photo.", "Super bad photo.")))
        (['Good photo.', 'Bad photo.', 'Super good photo.', 'Super bad photo.'], ['quality', 'user_defined_0'])

    """
    if not isinstance(prompts, tuple):
        raise ValueError('Argument `prompts` must be a tuple containing strings or tuples of strings')
    prompts_names: List[str] = []
    prompts_list: List[str] = []
    count = 0
    for p in prompts:
        if not isinstance(p, (str, tuple)):
            raise ValueError('Argument `prompts` must be a tuple containing strings or tuples of strings')
        if isinstance(p, str):
            if p not in _PROMPTS:
                raise ValueError(f'All elements of `prompts` must be one of {_PROMPTS.keys()} if not custom tuple prompts, got {p}.')
            prompts_names.append(p)
            prompts_list.extend(_PROMPTS[p])
        if isinstance(p, tuple) and len(p) != 2:
            raise ValueError('If a tuple is provided in argument `prompts`, it must be of length 2')
        if isinstance(p, tuple):
            prompts_names.append(f'user_defined_{count}')
            prompts_list.extend(p)
            count += 1
    return (prompts_list, prompts_names)