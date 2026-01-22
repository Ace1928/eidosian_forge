import collections
import itertools
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Type
import pandas as pd
import pyarrow as pa
import pyarrow.json as paj
import datasets
from datasets.features.features import FeatureType
from datasets.tasks.base import TaskTemplate
def _read_metadata(self, metadata_file, metadata_ext: str=''):
    if metadata_ext == '.csv':
        return pa.Table.from_pandas(pd.read_csv(metadata_file))
    else:
        with open(metadata_file, 'rb') as f:
            return paj.read_json(f)