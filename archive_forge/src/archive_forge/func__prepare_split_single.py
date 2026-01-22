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
from typing import TYPE_CHECKING, Dict, Iterable, Mapping, Optional, Tuple, Union
from unittest.mock import patch
import fsspec
import pyarrow as pa
from multiprocess import Pool
from tqdm.contrib.concurrent import thread_map
from . import config, utils
from .arrow_dataset import Dataset
from .arrow_reader import (
from .arrow_writer import ArrowWriter, BeamWriter, ParquetWriter, SchemaInferenceError
from .data_files import DataFilesDict, DataFilesPatternsDict, sanitize_patterns
from .dataset_dict import DatasetDict, IterableDatasetDict
from .download.download_config import DownloadConfig
from .download.download_manager import DownloadManager, DownloadMode
from .download.mock_download_manager import MockDownloadManager
from .download.streaming_download_manager import StreamingDownloadManager, xjoin, xopen
from .exceptions import DatasetGenerationCastError, DatasetGenerationError, FileFormatError, ManualDownloadError
from .features import Features
from .filesystems import (
from .fingerprint import Hasher
from .info import DatasetInfo, DatasetInfosDict, PostProcessedInfo
from .iterable_dataset import ArrowExamplesIterable, ExamplesIterable, IterableDataset
from .keyhash import DuplicatedKeysError
from .naming import INVALID_WINDOWS_CHARACTERS_IN_PATH, camelcase_to_snakecase
from .splits import Split, SplitDict, SplitGenerator, SplitInfo
from .streaming import extend_dataset_builder_for_streaming
from .table import CastError
from .utils import logging
from .utils import tqdm as hf_tqdm
from .utils._filelock import FileLock
from .utils.file_utils import cached_path, is_remote_url
from .utils.info_utils import VerificationMode, get_size_checksum_dict, verify_checksums, verify_splits
from .utils.py_utils import (
from .utils.sharding import _number_of_shards_in_gen_kwargs, _split_gen_kwargs
from .utils.track import tracked_list
def _prepare_split_single(self, gen_kwargs: dict, fpath: str, file_format: str, max_shard_size: int, job_id: int) -> Iterable[Tuple[int, bool, Union[int, tuple]]]:
    gen_kwargs = {k: tracked_list(v) if isinstance(v, list) else v for k, v in gen_kwargs.items()}
    generator = self._generate_tables(**gen_kwargs)
    writer_class = ParquetWriter if file_format == 'parquet' else ArrowWriter
    embed_local_files = file_format == 'parquet'
    shard_lengths = []
    total_num_examples, total_num_bytes = (0, 0)
    shard_id = 0
    num_examples_progress_update = 0
    try:
        writer = writer_class(features=self.info.features, path=fpath.replace('SSSSS', f'{shard_id:05d}').replace('JJJJJ', f'{job_id:05d}'), writer_batch_size=self._writer_batch_size, storage_options=self._fs.storage_options, embed_local_files=embed_local_files)
        try:
            _time = time.time()
            for _, table in generator:
                if max_shard_size is not None and writer._num_bytes > max_shard_size:
                    num_examples, num_bytes = writer.finalize()
                    writer.close()
                    shard_lengths.append(num_examples)
                    total_num_examples += num_examples
                    total_num_bytes += num_bytes
                    shard_id += 1
                    writer = writer_class(features=writer._features, path=fpath.replace('SSSSS', f'{shard_id:05d}').replace('JJJJJ', f'{job_id:05d}'), writer_batch_size=self._writer_batch_size, storage_options=self._fs.storage_options, embed_local_files=embed_local_files)
                try:
                    writer.write_table(table)
                except CastError as cast_error:
                    raise DatasetGenerationCastError.from_cast_error(cast_error=cast_error, builder_name=self.info.builder_name, gen_kwargs=gen_kwargs, token=self.token)
                num_examples_progress_update += len(table)
                if time.time() > _time + config.PBAR_REFRESH_TIME_INTERVAL:
                    _time = time.time()
                    yield (job_id, False, num_examples_progress_update)
                    num_examples_progress_update = 0
        finally:
            yield (job_id, False, num_examples_progress_update)
            num_shards = shard_id + 1
            num_examples, num_bytes = writer.finalize()
            writer.close()
            shard_lengths.append(num_examples)
            total_num_examples += num_examples
            total_num_bytes += num_bytes
    except Exception as e:
        if isinstance(e, SchemaInferenceError) and e.__context__ is not None:
            e = e.__context__
        if isinstance(e, DatasetGenerationError):
            raise
        raise DatasetGenerationError('An error occurred while generating the dataset') from e
    yield (job_id, True, (total_num_examples, total_num_bytes, writer._features, num_shards, shard_lengths))