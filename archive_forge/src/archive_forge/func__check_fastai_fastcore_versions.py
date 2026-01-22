import json
import os
from pathlib import Path
from pickle import DEFAULT_PROTOCOL, PicklingError
from typing import Any, Dict, List, Optional, Union
from packaging import version
from huggingface_hub import snapshot_download
from huggingface_hub.constants import CONFIG_NAME
from huggingface_hub.hf_api import HfApi
from huggingface_hub.utils import (
from .utils import logging, validate_hf_hub_args
from .utils._runtime import _PY_VERSION  # noqa: F401 # for backward compatibility...
def _check_fastai_fastcore_versions(fastai_min_version: str='2.4', fastcore_min_version: str='1.3.27'):
    """
    Checks that the installed fastai and fastcore versions are compatible for pickle serialization.

    Args:
        fastai_min_version (`str`, *optional*):
            The minimum fastai version supported.
        fastcore_min_version (`str`, *optional*):
            The minimum fastcore version supported.

    <Tip>
    Raises the following error:

        - [`ImportError`](https://docs.python.org/3/library/exceptions.html#ImportError)
          if the fastai or fastcore libraries are not available or are of an invalid version.

    </Tip>
    """
    if (get_fastcore_version() or get_fastai_version()) == 'N/A':
        raise ImportError(f'fastai>={fastai_min_version} and fastcore>={fastcore_min_version} are required. Currently using fastai=={get_fastai_version()} and fastcore=={get_fastcore_version()}.')
    current_fastai_version = version.Version(get_fastai_version())
    current_fastcore_version = version.Version(get_fastcore_version())
    if current_fastai_version < version.Version(fastai_min_version):
        raise ImportError(f'`push_to_hub_fastai` and `from_pretrained_fastai` require a fastai>={fastai_min_version} version, but you are using fastai version {get_fastai_version()} which is incompatible. Upgrade with `pip install fastai==2.5.6`.')
    if current_fastcore_version < version.Version(fastcore_min_version):
        raise ImportError(f'`push_to_hub_fastai` and `from_pretrained_fastai` require a fastcore>={fastcore_min_version} version, but you are using fastcore version {get_fastcore_version()} which is incompatible. Upgrade with `pip install fastcore==1.3.27`.')