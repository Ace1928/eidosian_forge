import sys
from typing import TYPE_CHECKING, Iterable, List, Optional, Union
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.util import _check_pyarrow_version
from ray.data.block import Block, BlockAccessor, BlockMetadata
from ray.data.datasource import Datasource, ReadTask
from ray.util.annotations import DeveloperAPI
@DeveloperAPI
class HuggingFaceDatasource(Datasource):
    """Hugging Face Dataset datasource, for reading from a
    `Hugging Face Datasets Dataset <https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset/>`_.
    This Datasource implements a streamed read using a
    single read task, most beneficial for a
    `Hugging Face Datasets IterableDataset <https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.IterableDataset/>`_
    or datasets which are too large to fit in-memory.
    For an in-memory Hugging Face Dataset (`datasets.Dataset`), use :meth:`~ray.data.from_huggingface`
    directly for faster performance.
    """

    def __init__(self, dataset: Union['datasets.Dataset', 'datasets.IterableDataset'], batch_size: int=4096):
        if TRANSFORMERS_IMPORT_ERROR is not None:
            raise TRANSFORMERS_IMPORT_ERROR
        self._dataset = dataset
        self._batch_size = batch_size

    def estimate_inmemory_data_size(self) -> Optional[int]:
        return self._dataset.dataset_size

    def get_read_tasks(self, parallelism: int) -> List[ReadTask]:
        _check_pyarrow_version()
        import numpy as np
        import pandas as pd
        import pyarrow

        def _read_dataset(dataset: 'datasets.IterableDataset') -> Iterable[Block]:
            for batch in dataset.with_format('arrow').iter(batch_size=self._batch_size):
                if not isinstance(batch, (pyarrow.Table, pd.DataFrame, dict, np.array)):
                    raise ValueError(f"Batch format {type(batch)} isn't supported. Only the following batch formats are supported: dict (corresponds to `None` in `dataset.with_format()`), pyarrow.Table, np.array, pd.DataFrame.")
                if isinstance(batch, np.ndarray):
                    batch = {'item': batch}
                if isinstance(batch, dict):
                    batch = pyarrow.Table.from_pydict(batch)
                block = BlockAccessor.for_block(batch).to_default()
                yield block
        meta = BlockMetadata(num_rows=None, size_bytes=None, schema=None, input_files=None, exec_stats=None)
        read_tasks: List[ReadTask] = [ReadTask(lambda hfds=self._dataset: _read_dataset(hfds), meta)]
        return read_tasks