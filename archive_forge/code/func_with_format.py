import contextlib
import copy
import fnmatch
import json
import math
import posixpath
import re
import warnings
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import fsspec
import numpy as np
from huggingface_hub import (
from . import config
from .arrow_dataset import PUSH_TO_HUB_WITHOUT_METADATA_CONFIGS_SPLIT_PATTERN_SHARDED, Dataset
from .features import Features
from .features.features import FeatureType
from .info import DatasetInfo, DatasetInfosDict
from .naming import _split_re
from .splits import NamedSplit, Split, SplitDict, SplitInfo
from .table import Table
from .tasks import TaskTemplate
from .utils import logging
from .utils.deprecation_utils import deprecated
from .utils.doc_utils import is_documented_by
from .utils.hub import list_files_info
from .utils.metadata import MetadataConfigs
from .utils.py_utils import asdict, glob_pattern_to_regex, string_to_dict
from .utils.typing import PathLike
def with_format(self, type: Optional[str]=None) -> 'IterableDatasetDict':
    """
        Return a dataset with the specified format.
        This method only supports the "torch" format for now.
        The format is set to all the datasets of the dataset dictionary.

        Args:
            type (`str`, *optional*, defaults to `None`):
                If set to "torch", the returned dataset
                will be a subclass of `torch.utils.data.IterableDataset` to be used in a `DataLoader`.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", streaming=True)
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> def encode(example):
        ...     return tokenizer(examples["text"], truncation=True, padding="max_length")
        >>> ds = ds.map(encode, batched=True, remove_columns=["text"])
        >>> ds = ds.with_format("torch")
        ```
        """
    return IterableDatasetDict({k: dataset.with_format(type=type) for k, dataset in self.items()})