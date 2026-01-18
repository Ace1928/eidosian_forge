import copy
import dataclasses
import json
import os
import posixpath
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Union
import fsspec
from huggingface_hub import DatasetCard, DatasetCardData
from . import config
from .features import Features, Value
from .splits import SplitDict
from .tasks import TaskTemplate, task_template_from_dict
from .utils import Version
from .utils.logging import get_logger
from .utils.py_utils import asdict, unique_values
def write_to_directory(self, metric_info_dir, pretty_print=False):
    """Write `MetricInfo` as JSON to `metric_info_dir`.
        Also save the license separately in LICENCE.
        If `pretty_print` is True, the JSON will be pretty-printed with the indent level of 4.

        Example:

        ```py
        >>> from datasets import load_metric
        >>> metric = load_metric("accuracy")
        >>> metric.info.write_to_directory("/path/to/directory/")
        ```
        """
    with open(os.path.join(metric_info_dir, config.METRIC_INFO_FILENAME), 'w', encoding='utf-8') as f:
        json.dump(asdict(self), f, indent=4 if pretty_print else None)
    if self.license:
        with open(os.path.join(metric_info_dir, config.LICENSE_FILENAME), 'w', encoding='utf-8') as f:
            f.write(self.license)