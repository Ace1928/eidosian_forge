import io
import pandas
from pandas.io.common import get_handle, stringify_path
from modin.core.execution.unidist.common import SignalActor, UnidistWrapper
from modin.core.execution.unidist.generic.io import UnidistIO
from modin.core.io import (
from modin.core.storage_formats.pandas.parsers import (
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.experimental.core.io import (
from modin.experimental.core.storage_formats.pandas.parsers import (
from ..dataframe import PandasOnUnidistDataframe
from ..partitioning import PandasOnUnidistDataframePartition
class PandasOnUnidistIO(UnidistIO):
    """Factory providing methods for performing I/O operations using pandas as storage format on unidist as engine."""
    frame_cls = PandasOnUnidistDataframe
    query_compiler_cls = PandasQueryCompiler
    build_args = dict(frame_partition_cls=PandasOnUnidistDataframePartition, query_compiler_cls=PandasQueryCompiler, frame_cls=PandasOnUnidistDataframe, base_io=UnidistIO)

    def __make_read(*classes, build_args=build_args):
        return type('', (UnidistWrapper, *classes), build_args).read

    def __make_write(*classes, build_args=build_args):
        return type('', (UnidistWrapper, *classes), build_args).write
    read_csv = __make_read(PandasCSVParser, CSVDispatcher)
    read_fwf = __make_read(PandasFWFParser, FWFDispatcher)
    read_json = __make_read(PandasJSONParser, JSONDispatcher)
    read_parquet = __make_read(PandasParquetParser, ParquetDispatcher)
    to_parquet = __make_write(ParquetDispatcher)
    read_feather = __make_read(PandasFeatherParser, FeatherDispatcher)
    read_sql = __make_read(PandasSQLParser, SQLDispatcher)
    to_sql = __make_write(SQLDispatcher)
    read_excel = __make_read(PandasExcelParser, ExcelDispatcher)
    read_csv_glob = __make_read(ExperimentalPandasCSVGlobParser, ExperimentalCSVGlobDispatcher)
    read_parquet_glob = __make_read(ExperimentalPandasParquetParser, ExperimentalGlobDispatcher)
    to_parquet_glob = __make_write(ExperimentalGlobDispatcher, build_args={**build_args, 'base_write': UnidistIO.to_parquet})
    read_json_glob = __make_read(ExperimentalPandasJsonParser, ExperimentalGlobDispatcher)
    to_json_glob = __make_write(ExperimentalGlobDispatcher, build_args={**build_args, 'base_write': UnidistIO.to_json})
    read_xml_glob = __make_read(ExperimentalPandasXmlParser, ExperimentalGlobDispatcher)
    to_xml_glob = __make_write(ExperimentalGlobDispatcher, build_args={**build_args, 'base_write': UnidistIO.to_xml})
    read_pickle_glob = __make_read(ExperimentalPandasPickleParser, ExperimentalGlobDispatcher)
    to_pickle_glob = __make_write(ExperimentalGlobDispatcher, build_args={**build_args, 'base_write': UnidistIO.to_pickle})
    read_custom_text = __make_read(ExperimentalCustomTextParser, ExperimentalCustomTextDispatcher)
    read_sql_distributed = __make_read(ExperimentalSQLDispatcher, build_args={**build_args, 'base_read': read_sql})
    del __make_read
    del __make_write

    @staticmethod
    def _to_csv_check_support(kwargs):
        """
        Check if parallel version of ``to_csv`` could be used.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed to ``.to_csv()``.

        Returns
        -------
        bool
            Whether parallel version of ``to_csv`` is applicable.
        """
        path_or_buf = kwargs['path_or_buf']
        compression = kwargs['compression']
        if not isinstance(path_or_buf, str):
            return False
        if 'r' in kwargs['mode'] and '+' in kwargs['mode']:
            return False
        if kwargs['encoding'] is not None:
            encoding = kwargs['encoding'].lower()
            if 'u' in encoding or 'utf' in encoding:
                if '16' in encoding or '32' in encoding:
                    return False
        if compression is None or not compression == 'infer':
            return False
        if any((path_or_buf.endswith(ext) for ext in ['.gz', '.bz2', '.zip', '.xz'])):
            return False
        return True

    @classmethod
    def to_csv(cls, qc, **kwargs):
        """
        Write records stored in the `qc` to a CSV file.

        Parameters
        ----------
        qc : BaseQueryCompiler
            The query compiler of the Modin dataframe that we want to run ``to_csv`` on.
        **kwargs : dict
            Parameters for ``pandas.to_csv(**kwargs)``.
        """
        kwargs['path_or_buf'] = stringify_path(kwargs['path_or_buf'])
        if not cls._to_csv_check_support(kwargs):
            return UnidistIO.to_csv(qc, **kwargs)
        signals = SignalActor.remote(len(qc._modin_frame._partitions) + 1)

        def func(df, **kw):
            """
            Dump a chunk of rows as csv, then save them to target maintaining order.

            Parameters
            ----------
            df : pandas.DataFrame
                A chunk of rows to write to a CSV file.
            **kw : dict
                Arguments to pass to ``pandas.to_csv(**kw)`` plus an extra argument
                `partition_idx` serving as chunk index to maintain rows order.
            """
            partition_idx = kw['partition_idx']
            csv_kwargs = kwargs.copy()
            if partition_idx != 0:
                if 'w' in csv_kwargs['mode']:
                    csv_kwargs['mode'] = csv_kwargs['mode'].replace('w', 'a')
                csv_kwargs['header'] = False
            path_or_buf = csv_kwargs['path_or_buf']
            is_binary = 'b' in csv_kwargs['mode']
            csv_kwargs['path_or_buf'] = io.BytesIO() if is_binary else io.StringIO()
            storage_options = csv_kwargs.pop('storage_options', None)
            df.to_csv(**csv_kwargs)
            csv_kwargs.update({'storage_options': storage_options})
            content = csv_kwargs['path_or_buf'].getvalue()
            csv_kwargs['path_or_buf'].close()
            UnidistWrapper.materialize(signals.wait.remote(partition_idx))
            with get_handle(path_or_buf, csv_kwargs['mode'] if is_binary else csv_kwargs['mode'] + 't', encoding=kwargs['encoding'], errors=kwargs['errors'], compression=kwargs['compression'], storage_options=kwargs.get('storage_options', None), is_text=not is_binary) as handles:
                handles.handle.write(content)
            UnidistWrapper.materialize(signals.send.remote(partition_idx + 1))
            return pandas.DataFrame()
        UnidistWrapper.materialize(signals.send.remote(0))
        qc._modin_frame._propagate_index_objs(axis=None)
        result = qc._modin_frame._partition_mgr_cls.map_axis_partitions(axis=1, partitions=qc._modin_frame._partitions, map_func=func, keep_partitioning=True, lengths=None, enumerate_partitions=True, max_retries=0)
        UnidistWrapper.materialize([part.list_of_blocks[0] for row in result for part in row])