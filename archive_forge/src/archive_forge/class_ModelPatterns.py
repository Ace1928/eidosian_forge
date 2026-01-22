import difflib
import json
import os
import re
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from datetime import date
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple, Union
import yaml
from ..models import auto as auto_module
from ..models.auto.configuration_auto import model_type_to_module_name
from ..utils import is_flax_available, is_tf_available, is_torch_available, logging
from . import BaseTransformersCLICommand
@dataclass
class ModelPatterns:
    """
    Holds the basic information about a new model for the add-new-model-like command.

    Args:
        model_name (`str`): The model name.
        checkpoint (`str`): The checkpoint to use for doc examples.
        model_type (`str`, *optional*):
            The model type, the identifier used internally in the library like `bert` or `xlm-roberta`. Will default to
            `model_name` lowercased with spaces replaced with minuses (-).
        model_lower_cased (`str`, *optional*):
            The lowercased version of the model name, to use for the module name or function names. Will default to
            `model_name` lowercased with spaces and minuses replaced with underscores.
        model_camel_cased (`str`, *optional*):
            The camel-cased version of the model name, to use for the class names. Will default to `model_name`
            camel-cased (with spaces and minuses both considered as word separators.
        model_upper_cased (`str`, *optional*):
            The uppercased version of the model name, to use for the constant names. Will default to `model_name`
            uppercased with spaces and minuses replaced with underscores.
        config_class (`str`, *optional*):
            The tokenizer class associated with this model. Will default to `"{model_camel_cased}Config"`.
        tokenizer_class (`str`, *optional*):
            The tokenizer class associated with this model (leave to `None` for models that don't use a tokenizer).
        image_processor_class (`str`, *optional*):
            The image processor class associated with this model (leave to `None` for models that don't use an image
            processor).
        feature_extractor_class (`str`, *optional*):
            The feature extractor class associated with this model (leave to `None` for models that don't use a feature
            extractor).
        processor_class (`str`, *optional*):
            The processor class associated with this model (leave to `None` for models that don't use a processor).
    """
    model_name: str
    checkpoint: str
    model_type: Optional[str] = None
    model_lower_cased: Optional[str] = None
    model_camel_cased: Optional[str] = None
    model_upper_cased: Optional[str] = None
    config_class: Optional[str] = None
    tokenizer_class: Optional[str] = None
    image_processor_class: Optional[str] = None
    feature_extractor_class: Optional[str] = None
    processor_class: Optional[str] = None

    def __post_init__(self):
        if self.model_type is None:
            self.model_type = self.model_name.lower().replace(' ', '-')
        if self.model_lower_cased is None:
            self.model_lower_cased = self.model_name.lower().replace(' ', '_').replace('-', '_')
        if self.model_camel_cased is None:
            words = self.model_name.split(' ')
            words = list(chain(*[w.split('-') for w in words]))
            words = [w[0].upper() + w[1:] for w in words]
            self.model_camel_cased = ''.join(words)
        if self.model_upper_cased is None:
            self.model_upper_cased = self.model_name.upper().replace(' ', '_').replace('-', '_')
        if self.config_class is None:
            self.config_class = f'{self.model_camel_cased}Config'