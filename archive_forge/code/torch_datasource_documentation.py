from typing import TYPE_CHECKING
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data.block import BlockMetadata
from ray.data.datasource.datasource import Datasource, ReadTask
from ray.util.annotations import DeveloperAPI
Torch datasource, for reading from `Torch
    datasets <https://pytorch.org/docs/stable/data.html/>`_.
    This datasource implements a streaming read using a single read task.
    