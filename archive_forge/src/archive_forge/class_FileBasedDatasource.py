import io
import pathlib
import posixpath
import warnings
from typing import (
import numpy as np
import ray
from ray._private.utils import _add_creatable_buckets_param_if_s3_uri
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.util import (
from ray.data.block import Block, BlockAccessor
from ray.data.context import DataContext
from ray.data.datasource.block_path_provider import BlockWritePathProvider
from ray.data.datasource.datasource import Datasource, ReadTask, WriteResult
from ray.data.datasource.file_meta_provider import (
from ray.data.datasource.filename_provider import (
from ray.data.datasource.partitioning import (
from ray.data.datasource.path_util import (
from ray.util.annotations import Deprecated, DeveloperAPI, PublicAPI
@DeveloperAPI
class FileBasedDatasource(Datasource):
    """File-based datasource for reading files.

    Don't use this class directly. Instead, subclass it and implement `_read_stream()`.
    """
    _WRITE_FILE_PER_ROW = False
    _FILE_EXTENSIONS: Optional[Union[str, List[str]]] = None
    _NUM_THREADS_PER_TASK = 0

    def __init__(self, paths: Union[str, List[str]], *, filesystem: Optional['pyarrow.fs.FileSystem']=None, schema: Optional[Union[type, 'pyarrow.lib.Schema']]=None, open_stream_args: Optional[Dict[str, Any]]=None, meta_provider: BaseFileMetadataProvider=DefaultFileMetadataProvider(), partition_filter: PathPartitionFilter=None, partitioning: Partitioning=None, ignore_missing_paths: bool=False, shuffle: Union[Literal['files'], None]=None, include_paths: bool=False, file_extensions: Optional[List[str]]=None):
        _check_pyarrow_version()
        self._schema = schema
        self._open_stream_args = open_stream_args
        self._meta_provider = meta_provider
        self._partition_filter = partition_filter
        self._partitioning = partitioning
        self._ignore_missing_paths = ignore_missing_paths
        self._include_paths = include_paths
        paths, self._filesystem = _resolve_paths_and_filesystem(paths, filesystem)
        paths, file_sizes = map(list, zip(*meta_provider.expand_paths(paths, self._filesystem, partitioning, ignore_missing_paths=ignore_missing_paths)))
        if ignore_missing_paths and len(paths) == 0:
            raise ValueError("None of the provided paths exist. The 'ignore_missing_paths' field is set to True.")
        self._supports_distributed_reads = not _is_local_scheme(paths)
        if not self._supports_distributed_reads and ray.util.client.ray.is_connected():
            raise ValueError("Because you're using Ray Client, read tasks scheduled on the Ray cluster can't access your local files. To fix this issue, store files in cloud storage or a distributed filesystem like NFS.")
        if self._partition_filter is not None:
            path_to_size = dict(zip(paths, file_sizes))
            paths = self._partition_filter(paths)
            file_sizes = [path_to_size[p] for p in paths]
            if len(paths) == 0:
                raise ValueError("No input files found to read. Please double check that 'partition_filter' field is set properly.")
        if file_extensions is not None:
            path_to_size = dict(zip(paths, file_sizes))
            paths = [p for p in paths if _has_file_extension(p, file_extensions)]
            file_sizes = [path_to_size[p] for p in paths]
            if len(paths) == 0:
                raise ValueError("No input files found to read. Please double check that 'file_extensions' field is set properly.")
        self._file_metadata_shuffler = None
        if shuffle == 'files':
            self._file_metadata_shuffler = np.random.default_rng()
        self._paths_ref = ray.put(paths)
        self._file_sizes_ref = ray.put(file_sizes)

    def _paths(self) -> List[str]:
        return ray.get(self._paths_ref)

    def _file_sizes(self) -> List[float]:
        return ray.get(self._file_sizes_ref)

    def estimate_inmemory_data_size(self) -> Optional[int]:
        total_size = 0
        for sz in self._file_sizes():
            if sz is not None:
                total_size += sz
        return total_size

    def get_read_tasks(self, parallelism: int) -> List[ReadTask]:
        import numpy as np
        ctx = DataContext.get_current()
        open_stream_args = self._open_stream_args
        partitioning = self._partitioning
        paths = self._paths()
        file_sizes = self._file_sizes()
        if self._file_metadata_shuffler is not None:
            files_metadata = list(zip(paths, file_sizes))
            shuffled_files_metadata = [files_metadata[i] for i in self._file_metadata_shuffler.permutation(len(files_metadata))]
            paths, file_sizes = list(map(list, zip(*shuffled_files_metadata)))
        read_stream = self._read_stream
        filesystem = _wrap_s3_serialization_workaround(self._filesystem)
        if open_stream_args is None:
            open_stream_args = {}
        open_input_source = self._open_input_source

        def read_files(read_paths: Iterable[str]) -> Iterable[Block]:
            nonlocal filesystem, open_stream_args, partitioning
            DataContext._set_current(ctx)
            fs = _unwrap_s3_serialization_workaround(filesystem)
            for read_path in read_paths:
                partitions: Dict[str, str] = {}
                if partitioning is not None:
                    parse = PathPartitionParser(partitioning)
                    partitions = parse(read_path)
                with _open_file_with_retry(read_path, lambda: open_input_source(fs, read_path, **open_stream_args)) as f:
                    for block in read_stream(f, read_path):
                        if partitions:
                            block = _add_partitions(block, partitions)
                        if self._include_paths:
                            block_accessor = BlockAccessor.for_block(block)
                            block = block_accessor.append_column('path', [read_path] * block_accessor.num_rows())
                        yield block

        def create_read_task_fn(read_paths, num_threads):

            def read_task_fn():
                nonlocal num_threads, read_paths
                if num_threads > 0:
                    if len(read_paths) < num_threads:
                        num_threads = len(read_paths)
                    logger.get_logger().debug(f'Reading {len(read_paths)} files with {num_threads} threads.')
                    yield from make_async_gen(iter(read_paths), read_files, num_workers=num_threads)
                else:
                    logger.get_logger().debug(f'Reading {len(read_paths)} files.')
                    yield from read_files(read_paths)
            return read_task_fn
        parallelism = min(parallelism, len(paths))
        read_tasks = []
        for read_paths, file_sizes in zip(np.array_split(paths, parallelism), np.array_split(file_sizes, parallelism)):
            if len(read_paths) <= 0:
                continue
            meta = self._meta_provider(read_paths, self._schema, rows_per_file=self._rows_per_file(), file_sizes=file_sizes)
            read_task_fn = create_read_task_fn(read_paths, self._NUM_THREADS_PER_TASK)
            read_task = ReadTask(read_task_fn, meta)
            read_tasks.append(read_task)
        return read_tasks

    def _open_input_source(self, filesystem: 'pyarrow.fs.FileSystem', path: str, **open_args) -> 'pyarrow.NativeFile':
        """Opens a source path for reading and returns the associated Arrow NativeFile.

        The default implementation opens the source path as a sequential input stream,
        using ctx.streaming_read_buffer_size as the buffer size if none is given by the
        caller.

        Implementations that do not support streaming reads (e.g. that require random
        access) should override this method.
        """
        import pyarrow as pa
        from pyarrow.fs import HadoopFileSystem
        compression = open_args.get('compression', None)
        if compression is None:
            try:
                compression = pa.Codec.detect(path).name
            except (ValueError, TypeError):
                import pathlib
                suffix = pathlib.Path(path).suffix
                if suffix and suffix[1:] == 'snappy':
                    compression = 'snappy'
                else:
                    compression = None
        buffer_size = open_args.pop('buffer_size', None)
        if buffer_size is None:
            ctx = DataContext.get_current()
            buffer_size = ctx.streaming_read_buffer_size
        if compression == 'snappy':
            open_args['compression'] = None
        else:
            open_args['compression'] = compression
        file = filesystem.open_input_stream(path, buffer_size=buffer_size, **open_args)
        if compression == 'snappy':
            import snappy
            stream = io.BytesIO()
            if isinstance(filesystem, HadoopFileSystem):
                snappy.hadoop_snappy.stream_decompress(src=file, dst=stream)
            else:
                snappy.stream_decompress(src=file, dst=stream)
            stream.seek(0)
            file = pa.PythonFile(stream, mode='r')
        return file

    def _rows_per_file(self):
        """Returns the number of rows per file, or None if unknown."""
        return None

    def _read_stream(self, f: 'pyarrow.NativeFile', path: str) -> Iterator[Block]:
        """Streaming read a single file.

        This method should be implemented by subclasses.
        """
        raise NotImplementedError('Subclasses of FileBasedDatasource must implement _read_stream().')

    def on_write_start(self, path: str, try_create_dir: bool=True, filesystem: Optional['pyarrow.fs.FileSystem']=None, **write_args) -> None:
        """Create a directory to write files to.

        If ``try_create_dir`` is ``False``, this method is a no-op.
        """
        from pyarrow.fs import FileType
        self.has_created_dir = False
        if try_create_dir:
            paths, filesystem = _resolve_paths_and_filesystem(path, filesystem)
            assert len(paths) == 1, len(paths)
            path = paths[0]
            if filesystem.get_file_info(path).type is FileType.NotFound:
                tmp = _add_creatable_buckets_param_if_s3_uri(path)
                filesystem.create_dir(tmp, recursive=True)
                self.has_created_dir = True

    def write(self, blocks: Iterable[Block], ctx: TaskContext, path: str, dataset_uuid: str, filesystem: Optional['pyarrow.fs.FileSystem']=None, try_create_dir: bool=True, open_stream_args: Optional[Dict[str, Any]]=None, block_path_provider: Optional[BlockWritePathProvider]=None, filename_provider: Optional[FilenameProvider]=None, write_args_fn: Callable[[], Dict[str, Any]]=lambda: {}, file_format: Optional[str]=None, _block_udf: Optional[Callable[[Block], Block]]=None, **write_args) -> WriteResult:
        """Write blocks for a file-based datasource."""
        if file_format is None:
            file_format = self._FILE_EXTENSIONS
            if isinstance(file_format, list):
                file_format = file_format[0]
        path, filesystem = _resolve_paths_and_filesystem(path, filesystem)
        path = path[0]
        _write_block_to_file = self._write_block
        _write_row_to_file = self._write_row
        if open_stream_args is None:
            open_stream_args = {}
        if block_path_provider is not None:
            warnings.warn('`block_path_provider` has been deprecated in favor of `filename_provider`. For more information, see https://docs.ray.io/en/master/data/api/doc/ray.data.datasource.FilenameProvider.html', DeprecationWarning)
        if filename_provider is None and block_path_provider is None:
            filename_provider = _DefaultFilenameProvider(dataset_uuid=dataset_uuid, file_format=file_format)
        num_rows_written = 0
        block_idx = 0
        for block in blocks:
            if _block_udf is not None:
                block = _block_udf(block)
            block = BlockAccessor.for_block(block)
            if block.num_rows() == 0:
                continue
            fs = _unwrap_s3_serialization_workaround(filesystem)
            if self._WRITE_FILE_PER_ROW:
                for row_idx, row in enumerate(block.iter_rows(public_row_format=False)):
                    if filename_provider is not None:
                        filename = filename_provider.get_filename_for_row(row, ctx.task_idx, block_idx, row_idx)
                    else:
                        filename = f'{dataset_uuid}_{ctx.task_idx:06}_{block_idx:06}_{row_idx:06}.{file_format}'
                    write_path = posixpath.join(path, filename)
                    logger.get_logger().debug(f'Writing {write_path} file.')
                    with _open_file_with_retry(write_path, lambda: fs.open_output_stream(write_path, **open_stream_args)) as f:
                        _write_row_to_file(f, row, writer_args_fn=write_args_fn, file_format=file_format, **write_args)
            else:
                if filename_provider is not None:
                    filename = filename_provider.get_filename_for_block(block, ctx.task_idx, block_idx)
                    write_path = posixpath.join(path, filename)
                else:
                    write_path = block_path_provider(path, filesystem=filesystem, dataset_uuid=dataset_uuid, task_index=ctx.task_idx, block_index=block_idx, file_format=file_format)
                logger.get_logger().debug(f'Writing {write_path} file.')
                with _open_file_with_retry(write_path, lambda: fs.open_output_stream(write_path, **open_stream_args)) as f:
                    _write_block_to_file(f, block, writer_args_fn=write_args_fn, **write_args)
            num_rows_written += block.num_rows()
            block_idx += 1
        if num_rows_written == 0:
            logger.get_logger().warning(f'Skipping writing empty dataset with UUID {dataset_uuid} at {path}')
            return 'skip'
        return 'ok'

    def on_write_complete(self, write_results: List[WriteResult], path: Optional[str]=None, filesystem: Optional['pyarrow.fs.FileSystem']=None, **kwargs) -> None:
        if not self.has_created_dir:
            return
        paths, filesystem = _resolve_paths_and_filesystem(path, filesystem)
        assert len(paths) == 1, len(paths)
        path = paths[0]
        if all((write_results == 'skip' for write_results in write_results)):
            filesystem.delete_dir(path)

    def _write_block(self, f: 'pyarrow.NativeFile', block: BlockAccessor, writer_args_fn: Callable[[], Dict[str, Any]]=lambda: {}, **writer_args):
        """Writes a block to a single file, passing all kwargs to the writer.

        This method should be implemented by subclasses.
        """
        raise NotImplementedError('Subclasses of FileBasedDatasource must implement _write_files().')

    def _write_row(self, f: 'pyarrow.NativeFile', row, writer_args_fn: Callable[[], Dict[str, Any]]=lambda: {}, **writer_args):
        """Writes a row to a single file, passing all kwargs to the writer.

        If `_WRITE_FILE_PER_ROW` is set to `True`, this method will be called instead
        of `_write_block()`.
        """
        raise NotImplementedError

    @property
    def supports_distributed_reads(self) -> bool:
        return self._supports_distributed_reads