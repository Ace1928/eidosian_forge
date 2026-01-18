import io
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Union
from tqdm.auto import tqdm as old_tqdm
from ..constants import HF_HUB_DISABLE_PROGRESS_BARS
Fix for https://github.com/huggingface/huggingface_hub/issues/1603