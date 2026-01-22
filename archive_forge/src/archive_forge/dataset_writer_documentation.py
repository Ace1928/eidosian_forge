import logging
import os
import time
from ray import data
from ray.rllib.offline.io_context import IOContext
from ray.rllib.offline.json_writer import _to_json_dict
from ray.rllib.offline.output_writer import OutputWriter
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.typing import SampleBatchType
from typing import Dict, List
Initializes a DatasetWriter instance.

        Examples:
        config = {
            "output": "dataset",
            "output_config": {
                "format": "json",
                "path": "/tmp/test_samples/",
                "max_num_samples_per_file": 100000,
            }
        }

        Args:
            ioctx: current IO context object.
            compress_columns: list of sample batch columns to compress.
        