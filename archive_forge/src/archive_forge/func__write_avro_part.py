from __future__ import annotations
import io
import uuid
from fsspec.core import OpenFile, get_fs_token_paths, open_files
from fsspec.utils import read_block
from fsspec.utils import tokenize as fs_tokenize
from dask.highlevelgraph import HighLevelGraph
def _write_avro_part(part, f, schema, codec, sync_interval, metadata):
    """Create single avro file from list of dictionaries"""
    import fastavro
    with f as f:
        fastavro.writer(f, schema, part, codec, sync_interval, metadata)