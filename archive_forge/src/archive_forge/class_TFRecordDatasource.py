import struct
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Optional, Union
import numpy as np
from ray.data.block import Block
from ray.data.datasource.file_based_datasource import FileBasedDatasource
from ray.util.annotations import PublicAPI
@PublicAPI(stability='alpha')
class TFRecordDatasource(FileBasedDatasource):
    """TFRecord datasource, for reading and writing TFRecord files."""
    _FILE_EXTENSIONS = ['tfrecords']

    def __init__(self, paths: Union[str, List[str]], tf_schema: Optional['schema_pb2.Schema']=None, **file_based_datasource_kwargs):
        super().__init__(paths, **file_based_datasource_kwargs)
        self.tf_schema = tf_schema

    def _read_stream(self, f: 'pyarrow.NativeFile', path: str) -> Iterator[Block]:
        import pyarrow as pa
        import tensorflow as tf
        from google.protobuf.message import DecodeError
        for record in _read_records(f, path):
            example = tf.train.Example()
            try:
                example.ParseFromString(record)
            except DecodeError as e:
                raise ValueError(f"`TFRecordDatasource` failed to parse `tf.train.Example` record in '{path}'. This error can occur if your TFRecord file contains a message type other than `tf.train.Example`: {e}")
            yield pa.Table.from_pydict(_convert_example_to_dict(example, self.tf_schema))