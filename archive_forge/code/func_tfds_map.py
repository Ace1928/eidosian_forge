import csv
import dataclasses
import json
from dataclasses import dataclass
from typing import List, Optional, Union
from ...utils import is_tf_available, is_torch_available, logging
def tfds_map(self, example):
    """
        Some tensorflow_datasets datasets are not formatted the same way the GLUE datasets are. This method converts
        examples to the correct format.
        """
    if len(self.get_labels()) > 1:
        example.label = self.get_labels()[int(example.label)]
    return example