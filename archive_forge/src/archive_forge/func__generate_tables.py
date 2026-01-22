import glob
import os
import shutil
import time
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union
import pyarrow as pa
import datasets
import datasets.config
import datasets.data_files
from datasets.naming import filenames_for_dataset_split
def _generate_tables(self, files):
    for file_idx, file in enumerate(files):
        with open(file, 'rb') as f:
            try:
                for batch_idx, record_batch in enumerate(pa.ipc.open_stream(f)):
                    pa_table = pa.Table.from_batches([record_batch])
                    yield (f'{file_idx}_{batch_idx}', pa_table)
            except ValueError as e:
                logger.error(f"Failed to read file '{file}' with error {type(e)}: {e}")
                raise