import sys
from typing import TYPE_CHECKING, Iterable, List, Optional, Union
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.util import _check_pyarrow_version
from ray.data.block import Block, BlockAccessor, BlockMetadata
from ray.data.datasource import Datasource, ReadTask
from ray.util.annotations import DeveloperAPI
Hugging Face Dataset datasource, for reading from a
    `Hugging Face Datasets Dataset <https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset/>`_.
    This Datasource implements a streamed read using a
    single read task, most beneficial for a
    `Hugging Face Datasets IterableDataset <https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.IterableDataset/>`_
    or datasets which are too large to fit in-memory.
    For an in-memory Hugging Face Dataset (`datasets.Dataset`), use :meth:`~ray.data.from_huggingface`
    directly for faster performance.
    