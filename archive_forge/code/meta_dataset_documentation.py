import hashlib
import json
from typing import Any, Dict, Optional
from mlflow.data.dataset import Dataset
from mlflow.data.dataset_source import DatasetSource
from mlflow.types import Schema
from mlflow.utils.annotations import experimental
Create config dictionary for the MetaDataset.

        Returns a string dictionary containing the following fields: name, digest, source, source
        type, schema, and profile.
        