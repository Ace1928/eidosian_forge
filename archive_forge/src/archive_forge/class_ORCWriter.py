from numbers import Integral
import warnings
from pyarrow.lib import Table
import pyarrow._orc as _orc
from pyarrow.fs import _resolve_filesystem_and_path
class ORCWriter:
    __doc__ = '\nWriter interface for a single ORC file\n\nParameters\n----------\nwhere : str or pyarrow.io.NativeFile\n    Writable target. For passing Python file objects or byte buffers,\n    see pyarrow.io.PythonFileInterface, pyarrow.io.BufferOutputStream\n    or pyarrow.io.FixedSizeBufferWriter.\n{}\n'.format(_orc_writer_args_docs)
    is_open = False

    def __init__(self, where, *, file_version='0.12', batch_size=1024, stripe_size=64 * 1024 * 1024, compression='uncompressed', compression_block_size=65536, compression_strategy='speed', row_index_stride=10000, padding_tolerance=0.0, dictionary_key_size_threshold=0.0, bloom_filter_columns=None, bloom_filter_fpp=0.05):
        self.writer = _orc.ORCWriter()
        self.writer.open(where, file_version=file_version, batch_size=batch_size, stripe_size=stripe_size, compression=compression, compression_block_size=compression_block_size, compression_strategy=compression_strategy, row_index_stride=row_index_stride, padding_tolerance=padding_tolerance, dictionary_key_size_threshold=dictionary_key_size_threshold, bloom_filter_columns=bloom_filter_columns, bloom_filter_fpp=bloom_filter_fpp)
        self.is_open = True

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def write(self, table):
        """
        Write the table into an ORC file. The schema of the table must
        be equal to the schema used when opening the ORC file.

        Parameters
        ----------
        table : pyarrow.Table
            The table to be written into the ORC file
        """
        assert self.is_open
        self.writer.write(table)

    def close(self):
        """
        Close the ORC file
        """
        if self.is_open:
            self.writer.close()
            self.is_open = False