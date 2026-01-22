import abc
import contextlib
import copy
import inspect
import os
import posixpath
import shutil
import textwrap
import time
import urllib
import warnings
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple, Union
import fsspec
from multiprocess import Pool
from tqdm.contrib.concurrent import thread_map
from datasets import ReadInstruction
from . import config, utils
from .arrow_dataset import Dataset
from .arrow_reader import (
from .arrow_writer import ArrowWriter, BeamWriter, ParquetWriter, SchemaInferenceError
from .data_files import DataFilesDict, sanitize_patterns
from .dataset_dict import DatasetDict, IterableDatasetDict
from .download.download_config import DownloadConfig
from .download.download_manager import DownloadManager, DownloadMode
from .download.mock_download_manager import MockDownloadManager
from .download.streaming_download_manager import StreamingDownloadManager
from .features import Features
from .filesystems import is_remote_filesystem
from .fingerprint import Hasher
from .info import DatasetInfo, DatasetInfosDict, PostProcessedInfo
from .iterable_dataset import ExamplesIterable, IterableDataset, _generate_examples_from_tables_wrapper
from .keyhash import DuplicatedKeysError
from .naming import INVALID_WINDOWS_CHARACTERS_IN_PATH, camelcase_to_snakecase
from .splits import Split, SplitDict, SplitGenerator, SplitInfo
from .streaming import extend_dataset_builder_for_streaming
from .utils import logging
from .utils.file_utils import cached_path, is_remote_url
from .utils.filelock import FileLock
from .utils.info_utils import VerificationMode, get_size_checksum_dict, verify_checksums, verify_splits
from .utils.py_utils import (
from .utils.sharding import _number_of_shards_in_gen_kwargs, _split_gen_kwargs
def as_streaming_dataset_original(self, split: Optional[str]=None, base_path: Optional[str]=None) -> Union[Dict[str, IterableDataset], IterableDataset]:
    relative_data_dir = self._relative_data_dir(with_version=True, with_hash=False)
    remote_cache_dir = HF_GCP_BASE_URL + '/' + relative_data_dir.replace(os.sep, '/')
    try:
        remote_dataset_info = f'{remote_cache_dir}/dataset_info.json'
        from .download.streaming_download_manager import xopen
        with xopen(remote_dataset_info) as f:
            import json
            _info = json.load(f)
        if self.info is not None:
            self.info.update(DatasetInfo.from_dict(_info))
    except FileNotFoundError as err:
        raise DatasetNotOnHfGcsError(err) from None
    try:
        for split in self.info.splits:
            import pdb
            pdb.set_trace()
            file_instructions = [{'filename': f'{self.name}-{split}.arrow'}]
            for file_instruction in file_instructions:
                remote_prepared_filename = os.path.join(remote_cache_dir, file_instruction['filename'])
                import pyarrow as pa
                with xopen(remote_prepared_filename, 'rb') as f:
                    with pa.ipc.open_stream(f) as reader:
                        for record_batch in reader:
                            yield from record_batch.to_pylist()
    except FileNotFoundError as err:
        raise MissingFilesOnHfGcsError(err) from None