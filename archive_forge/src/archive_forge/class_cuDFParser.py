import warnings
from io import BytesIO
import numpy as np
import pandas
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.concat import union_categoricals
from pandas.io.common import infer_compression
from modin.config import MinPartitionSize
from modin.core.execution.ray.implementations.cudf_on_ray.partitioning.partition_manager import (
from modin.core.io.file_dispatcher import OpenFile
from modin.core.storage_formats.pandas.utils import split_result_of_axis_func_pandas
from modin.error_message import ErrorMessage
class cuDFParser(object):

    @classmethod
    def get_dtypes(cls, dtypes_ids):
        return pandas.concat(cls.materialize(dtypes_ids), axis=1).apply(lambda row: find_common_type_cat(row.values), axis=1).squeeze(axis=0)

    @classmethod
    def single_worker_read(cls, fname, *, reason, **kwargs):
        ErrorMessage.default_to_pandas(reason=reason)
        pandas_frame = cls.parse(fname, **kwargs)
        if isinstance(pandas_frame, pandas.io.parsers.TextFileReader):
            pd_read = pandas_frame.read
            pandas_frame.read = lambda *args, **kwargs: cls.query_compiler_cls.from_pandas(pd_read(*args, **kwargs), cls.frame_cls)
            return pandas_frame
        elif isinstance(pandas_frame, dict):
            return {i: cls.query_compiler_cls.from_pandas(frame, cls.frame_cls) for i, frame in pandas_frame.items()}
        return cls.query_compiler_cls.from_pandas(pandas_frame, cls.frame_cls)
    infer_compression = infer_compression