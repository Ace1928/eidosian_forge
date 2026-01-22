from numbers import Integral
import warnings
from pyarrow.lib import Table
import pyarrow._orc as _orc
from pyarrow.fs import _resolve_filesystem_and_path
class ORCFile:
    """
    Reader interface for a single ORC file

    Parameters
    ----------
    source : str or pyarrow.NativeFile
        Readable source. For passing Python file objects or byte buffers,
        see pyarrow.io.PythonFileInterface or pyarrow.io.BufferReader.
    """

    def __init__(self, source):
        self.reader = _orc.ORCReader()
        self.reader.open(source)

    @property
    def metadata(self):
        """The file metadata, as an arrow KeyValueMetadata"""
        return self.reader.metadata()

    @property
    def schema(self):
        """The file schema, as an arrow schema"""
        return self.reader.schema()

    @property
    def nrows(self):
        """The number of rows in the file"""
        return self.reader.nrows()

    @property
    def nstripes(self):
        """The number of stripes in the file"""
        return self.reader.nstripes()

    @property
    def file_version(self):
        """Format version of the ORC file, must be 0.11 or 0.12"""
        return self.reader.file_version()

    @property
    def software_version(self):
        """Software instance and version that wrote this file"""
        return self.reader.software_version()

    @property
    def compression(self):
        """Compression codec of the file"""
        return self.reader.compression()

    @property
    def compression_size(self):
        """Number of bytes to buffer for the compression codec in the file"""
        return self.reader.compression_size()

    @property
    def writer(self):
        """Name of the writer that wrote this file.
        If the writer is unknown then its Writer ID
        (a number) is returned"""
        return self.reader.writer()

    @property
    def writer_version(self):
        """Version of the writer"""
        return self.reader.writer_version()

    @property
    def row_index_stride(self):
        """Number of rows per an entry in the row index or 0
        if there is no row index"""
        return self.reader.row_index_stride()

    @property
    def nstripe_statistics(self):
        """Number of stripe statistics"""
        return self.reader.nstripe_statistics()

    @property
    def content_length(self):
        """Length of the data stripes in the file in bytes"""
        return self.reader.content_length()

    @property
    def stripe_statistics_length(self):
        """The number of compressed bytes in the file stripe statistics"""
        return self.reader.stripe_statistics_length()

    @property
    def file_footer_length(self):
        """The number of compressed bytes in the file footer"""
        return self.reader.file_footer_length()

    @property
    def file_postscript_length(self):
        """The number of bytes in the file postscript"""
        return self.reader.file_postscript_length()

    @property
    def file_length(self):
        """The number of bytes in the file"""
        return self.reader.file_length()

    def _select_names(self, columns=None):
        if columns is None:
            return None
        schema = self.schema
        names = []
        for col in columns:
            if isinstance(col, Integral):
                col = int(col)
                if 0 <= col < len(schema):
                    col = schema[col].name
                    names.append(col)
                else:
                    raise ValueError('Column indices must be in 0 <= ind < %d, got %d' % (len(schema), col))
            else:
                return columns
        return names

    def read_stripe(self, n, columns=None):
        """Read a single stripe from the file.

        Parameters
        ----------
        n : int
            The stripe index
        columns : list
            If not None, only these columns will be read from the stripe. A
            column name may be a prefix of a nested field, e.g. 'a' will select
            'a.b', 'a.c', and 'a.d.e'

        Returns
        -------
        pyarrow.RecordBatch
            Content of the stripe as a RecordBatch.
        """
        columns = self._select_names(columns)
        return self.reader.read_stripe(n, columns=columns)

    def read(self, columns=None):
        """Read the whole file.

        Parameters
        ----------
        columns : list
            If not None, only these columns will be read from the file. A
            column name may be a prefix of a nested field, e.g. 'a' will select
            'a.b', 'a.c', and 'a.d.e'. Output always follows the
            ordering of the file and not the `columns` list.

        Returns
        -------
        pyarrow.Table
            Content of the file as a Table.
        """
        columns = self._select_names(columns)
        return self.reader.read(columns=columns)