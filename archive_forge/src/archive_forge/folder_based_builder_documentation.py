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

    Base class for generic data loaders for vision and image data.


    Abstract class attributes to be overridden by a child class:
        BASE_FEATURE: feature object to decode data (i.e. datasets.Image, datasets.Audio, ...)
        BASE_COLUMN_NAME: string key name of a base feature (i.e. "image", "audio", ...)
        BUILDER_CONFIG_CLASS: builder config inherited from `folder_based_builder.FolderBasedBuilderConfig`
        EXTENSIONS: list of allowed extensions (only files with these extensions and METADATA_FILENAME files
            will be included in a dataset)
        CLASSIFICATION_TASK: classification task to use if labels are obtained from the folder structure
    