import fnmatch
import io
import re
import tarfile
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
from ray.data.block import BlockAccessor
from ray.data.datasource.file_based_datasource import FileBasedDatasource
from ray.util.annotations import PublicAPI
def _apply_list(f: Union[Callable, List[Callable]], sample: Dict[str, Any], default: Callable=None):
    """Apply a list of functions to a sample.

    Args:
        f: function or list of functions
        sample: sample to be modified
        default: default function to be applied to all keys.
            Defaults to None.

    Returns:
        modified sample
    """
    if f is None:
        return sample
    if not isinstance(f, list):
        f = [f]
    for g in f:
        if default is not None and (not callable(g)):
            g = partial(default, format=g)
        sample = g(sample)
    return sample