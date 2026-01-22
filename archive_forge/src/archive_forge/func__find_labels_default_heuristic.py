from __future__ import annotations
import collections.abc
import numbers
from contextlib import suppress
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Type, Union
import PIL.Image
import torch
from torchvision import tv_tensors
from torchvision._utils import sequence_to_str
from torchvision.transforms.transforms import _check_sequence_input, _setup_angle, _setup_size  # noqa: F401
from torchvision.transforms.v2.functional import get_dimensions, get_size, is_pure_tensor
from torchvision.transforms.v2.functional._utils import _FillType, _FillTypeJIT
def _find_labels_default_heuristic(inputs: Any) -> torch.Tensor:
    """
    This heuristic covers three cases:

    1. The input is tuple or list whose second item is a labels tensor. This happens for already batched
       classification inputs for MixUp and CutMix (typically after the Dataloder).
    2. The input is a tuple or list whose second item is a dictionary that contains the labels tensor
       under a label-like (see below) key. This happens for the inputs of detection models.
    3. The input is a dictionary that is structured as the one from 2.

    What is "label-like" key? We first search for an case-insensitive match of 'labels' inside the keys of the
    dictionary. This is the name our detection models expect. If we can't find that, we look for a case-insensitive
    match of the term 'label' anywhere inside the key, i.e. 'FooLaBeLBar'. If we can't find that either, the dictionary
    contains no "label-like" key.
    """
    if isinstance(inputs, (tuple, list)):
        inputs = inputs[1]
    if is_pure_tensor(inputs):
        return inputs
    if not isinstance(inputs, collections.abc.Mapping):
        raise ValueError(f'When using the default labels_getter, the input passed to forward must be a dictionary or a two-tuple whose second item is a dictionary or a tensor, but got {inputs} instead.')
    candidate_key = None
    with suppress(StopIteration):
        candidate_key = next((key for key in inputs.keys() if key.lower() == 'labels'))
    if candidate_key is None:
        with suppress(StopIteration):
            candidate_key = next((key for key in inputs.keys() if 'label' in key.lower()))
    if candidate_key is None:
        raise ValueError('Could not infer where the labels are in the sample. Try passing a callable as the labels_getter parameter?If there are no labels in the sample by design, pass labels_getter=None.')
    return inputs[candidate_key]