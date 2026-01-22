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
class ArrowBasedBuilder(DatasetBuilder):
    """Base class for datasets with data generation based on Arrow loading functions (CSV/JSON/Parquet)."""

    @abc.abstractmethod
    def _generate_tables(self, **kwargs):
        """Default function generating examples for each `SplitGenerator`.

        This function preprocess the examples from the raw data to the preprocessed
        dataset files.
        This function is called once for each `SplitGenerator` defined in
        `_split_generators`. The examples yielded here will be written on
        disk.

        Args:
            **kwargs (additional keyword arguments):
                Arguments forwarded from the SplitGenerator.gen_kwargs

        Yields:
            key: `str` or `int`, a unique deterministic example identification key.
                * Unique: An error will be raised if two examples are yield with the
                    same key.
                * Deterministic: When generating the dataset twice, the same example
                    should have the same key.
                Good keys can be the image id, or line number if examples are extracted
                from a text file.
                The key will be hashed and sorted to shuffle examples deterministically,
                such as generating the dataset multiple times keep examples in the
                same order.
            example: `pyarrow.Table`, a feature table
                ready to be encoded and written to disk.
        """
        raise NotImplementedError()

    def _prepare_split(self, split_generator: SplitGenerator, file_format: str='arrow', num_proc: Optional[int]=None, max_shard_size: Optional[Union[str, int]]=None):
        max_shard_size = convert_file_size_to_int(max_shard_size or config.MAX_SHARD_SIZE)
        try:
            split_info = self.info.splits[split_generator.name]
        except Exception:
            split_info = split_generator.split_info
        SUFFIX = '-JJJJJ-SSSSS-of-NNNNN'
        fname = f'{self.dataset_name}-{split_generator.name}{SUFFIX}.{file_format}'
        fpath = posixpath.join(self._output_dir, fname)
        if num_proc and num_proc > 1:
            num_input_shards = _number_of_shards_in_gen_kwargs(split_generator.gen_kwargs)
            if num_input_shards <= 1:
                logger.warning(f'Setting num_proc from {num_proc} back to 1 for the {split_info.name} split to disable multiprocessing as it only contains one shard.')
                num_proc = 1
            elif num_input_shards < num_proc:
                logger.warning(f'Setting num_proc from {num_proc} to {num_input_shards} for the {split_info.name} split as it only contains {num_input_shards} shards.')
                num_proc = num_input_shards
        pbar = hf_tqdm(unit=' examples', total=split_info.num_examples, desc=f'Generating {split_info.name} split')
        _prepare_split_args = {'fpath': fpath, 'file_format': file_format, 'max_shard_size': max_shard_size}
        if num_proc is None or num_proc == 1:
            result = None
            gen_kwargs = split_generator.gen_kwargs
            job_id = 0
            with pbar:
                for job_id, done, content in self._prepare_split_single(gen_kwargs=gen_kwargs, job_id=job_id, **_prepare_split_args):
                    if done:
                        result = content
                    else:
                        pbar.update(content)
            assert result is not None, 'Failed to retrieve results from prepare_split'
            examples_per_job, bytes_per_job, features_per_job, shards_per_job, shard_lengths_per_job = [[item] for item in result]
        else:
            kwargs_per_job = [{'gen_kwargs': gen_kwargs, 'job_id': job_id, **_prepare_split_args} for job_id, gen_kwargs in enumerate(_split_gen_kwargs(split_generator.gen_kwargs, max_num_jobs=num_proc))]
            num_jobs = len(kwargs_per_job)
            examples_per_job = [None] * num_jobs
            bytes_per_job = [None] * num_jobs
            features_per_job = [None] * num_jobs
            shards_per_job = [None] * num_jobs
            shard_lengths_per_job = [None] * num_jobs
            with Pool(num_proc) as pool:
                with pbar:
                    for job_id, done, content in iflatmap_unordered(pool, self._prepare_split_single, kwargs_iterable=kwargs_per_job):
                        if done:
                            examples_per_job[job_id], bytes_per_job[job_id], features_per_job[job_id], shards_per_job[job_id], shard_lengths_per_job[job_id] = content
                        else:
                            pbar.update(content)
            assert None not in examples_per_job, f'Failed to retrieve results from prepare_split: result list {examples_per_job} still contains None - at least one worker failed to return its results'
        total_shards = sum(shards_per_job)
        total_num_examples = sum(examples_per_job)
        total_num_bytes = sum(bytes_per_job)
        features = features_per_job[0]
        split_generator.split_info.num_examples = total_num_examples
        split_generator.split_info.num_bytes = total_num_bytes
        logger.debug(f'Renaming {total_shards} shards.')
        if total_shards > 1:

            def _rename_shard(shard_id_and_job: Tuple[int]):
                shard_id, job_id = shard_id_and_job
                global_shard_id = sum(shards_per_job[:job_id]) + shard_id
                self._rename(fpath.replace('SSSSS', f'{shard_id:05d}').replace('JJJJJ', f'{job_id:05d}'), fpath.replace('JJJJJ-SSSSS', f'{global_shard_id:05d}').replace('NNNNN', f'{total_shards:05d}'))
            shard_ids_and_jobs = [(shard_id, job_id) for job_id, num_shards in enumerate(shards_per_job) for shard_id in range(num_shards)]
            thread_map(_rename_shard, shard_ids_and_jobs, disable=True, max_workers=64)
            split_generator.split_info.shard_lengths = [shard_length for shard_lengths in shard_lengths_per_job for shard_length in shard_lengths]
        else:
            shard_id, job_id = (0, 0)
            self._rename(fpath.replace('SSSSS', f'{shard_id:05d}').replace('JJJJJ', f'{job_id:05d}'), fpath.replace(SUFFIX, ''))
        if self.info.features is None:
            self.info.features = features

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

    def _get_examples_iterable_for_split(self, split_generator: SplitGenerator) -> ExamplesIterable:
        return ArrowExamplesIterable(self._generate_tables, kwargs=split_generator.gen_kwargs)