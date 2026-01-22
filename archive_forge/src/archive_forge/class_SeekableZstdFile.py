from array import array
from bisect import bisect_right
from os.path import isfile
from struct import Struct
from warnings import warn
from pyzstd.zstdfile import ZstdDecompressReader, ZstdFile, \
class SeekableZstdFile(ZstdFile):
    """This class can only create/write/read Zstandard Seekable Format file,
    or read 0-size file.
    It provides relatively fast seeking ability in read mode.
    """
    FRAME_MAX_C_SIZE = 2 * 1024 * 1024 * 1024
    FRAME_MAX_D_SIZE = 1 * 1024 * 1024 * 1024
    _READER_CLASS = SeekableDecompressReader

    def __init__(self, filename, mode='r', *, level_or_option=None, zstd_dict=None, read_size=131075, write_size=131591, max_frame_content_size=1024 * 1024 * 1024):
        """Open a Zstandard Seekable Format file in binary mode. In read mode,
        the file can be 0-size file.

        filename can be either an actual file name (given as a str, bytes, or
        PathLike object), in which case the named file is opened, or it can be
        an existing file object to read from or write to.

        mode can be "r" for reading (default), "w" for (over)writing, "x" for
        creating exclusively, or "a" for appending. These can equivalently be
        given as "rb", "wb", "xb" and "ab" respectively.

        In append mode ("a" or "ab"), filename argument can't be a file object,
        please use file path.

        Parameters
        level_or_option: When it's an int object, it represents compression
            level. When it's a dict object, it contains advanced compression
            parameters. Note, in read mode (decompression), it can only be a
            dict object, that represents decompression option. It doesn't
            support int type compression level in this case.
        zstd_dict: A ZstdDict object, pre-trained dictionary for compression /
            decompression.
        read_size: In reading mode, this is bytes number that read from the
            underlying file object each time, default value is zstd's
            recommended value. If use with Network File System, increasing
            it may get better performance.
        write_size: In writing modes, this is output buffer's size, default
            value is zstd's recommended value. If use with Network File
            System, increasing it may get better performance.
        max_frame_content_size: In write/append modes (compression), when
            the uncompressed data size reaches max_frame_content_size, a frame
            is generated automatically. If the size is small, it will increase
            seeking speed, but reduce compression ratio. If the size is large,
            it will reduce seeking speed, but increase compression ratio. You
            can also manually generate a frame using f.flush(f.FLUSH_FRAME).
        """
        self._write_in_close = False
        self._fp = None
        self._closefp = False
        self._mode = _MODE_CLOSED
        if mode in ('r', 'rb'):
            if max_frame_content_size != 1024 * 1024 * 1024:
                raise ValueError('max_frame_content_size argument is only valid in write modes (compression).')
        elif mode in ('w', 'wb', 'a', 'ab', 'x', 'xb'):
            if not 0 < max_frame_content_size <= self.FRAME_MAX_D_SIZE:
                raise ValueError('max_frame_content_size argument should be 0 < value <= %d, provided value is %d.' % (self.FRAME_MAX_D_SIZE, max_frame_content_size))
            self._max_frame_content_size = max_frame_content_size
            self._reset_frame_sizes()
            self._seek_table = SeekTable(read_mode=False)
            if mode in ('a', 'ab'):
                if not isinstance(filename, (str, bytes, PathLike)):
                    raise TypeError("In append mode ('a', 'ab'), SeekableZstdFile.__init__() method can't accept file object as filename argument. Please use file path (str/bytes/PathLike).")
                if isfile(filename):
                    with io.open(filename, 'rb') as f:
                        if not hasattr(f, 'seekable') or not f.seekable():
                            raise TypeError("In SeekableZstdFile's append mode ('a', 'ab'), the opened 'rb' file object should be seekable.")
                        self._seek_table.load_seek_table(f, seek_to_0=False)
        super().__init__(filename, mode, level_or_option=level_or_option, zstd_dict=zstd_dict, read_size=read_size, write_size=write_size)
        if mode in ('a', 'ab'):
            if self._fp.seekable():
                self._fp.seek(self._seek_table.get_full_c_size())
                self._fp.truncate()
            else:
                self._seek_table.append_entry(self._seek_table.seek_frame_size, 0)
                warn("SeekableZstdFile is opened in append mode ('a', 'ab'), but the underlying file object is not seekable. Therefore the seek table (a zstd skippable frame) at the end of the file can't be overwritten. Each time open such file in append mode, it will waste some storage space. %d bytes were wasted this time." % self._seek_table.seek_frame_size, RuntimeWarning, 2)
        self._write_in_close = self._mode == _MODE_WRITE

    def _reset_frame_sizes(self):
        self._current_c_size = 0
        self._current_d_size = 0
        self._left_d_size = self._max_frame_content_size

    def close(self):
        """Flush and close the file.

        May be called more than once without error. Once the file is
        closed, any other operation on it will raise a ValueError.
        """
        try:
            if self._write_in_close:
                try:
                    self.flush(self.FLUSH_FRAME)
                    self._seek_table.write_seek_table(self._fp)
                finally:
                    self._write_in_close = False
        finally:
            self._seek_table = None
            super().close()

    def write(self, data):
        """Write a bytes-like object to the file.

        Returns the number of uncompressed bytes written, which is
        always the length of data in bytes. Note that due to buffering,
        the file on disk may not reflect the data written until .flush()
        or .close() is called.
        """
        if self._mode != _MODE_WRITE:
            self._check_mode(_MODE_WRITE)
        with memoryview(data) as view, view.cast('B') as byte_view:
            nbytes = byte_view.nbytes
            pos = 0
            while nbytes > 0:
                write_size = min(nbytes, self._left_d_size)
                _, output_size = self._writer.write(byte_view[pos:pos + write_size])
                self._pos += write_size
                pos += write_size
                nbytes -= write_size
                self._current_c_size += output_size
                self._current_d_size += write_size
                self._left_d_size -= write_size
                if self._left_d_size == 0 or self._current_c_size >= self.FRAME_MAX_C_SIZE:
                    self.flush(self.FLUSH_FRAME)
            return pos

    def flush(self, mode=ZstdFile.FLUSH_BLOCK):
        """Flush remaining data to the underlying stream.

        The mode argument can be ZstdFile.FLUSH_BLOCK, ZstdFile.FLUSH_FRAME.
        Abuse of this method will reduce compression ratio, use it only when
        necessary.

        If the program is interrupted afterwards, all data can be recovered.
        To ensure saving to disk, also need to use os.fsync(fd).

        This method does nothing in reading mode.
        """
        if self._mode != _MODE_WRITE:
            if self._mode == _MODE_READ:
                return
            self._check_mode()
        _, output_size = self._writer.flush(mode)
        self._current_c_size += output_size
        if mode == self.FLUSH_FRAME and self._current_c_size != 0:
            self._seek_table.append_entry(self._current_c_size, self._current_d_size)
            self._reset_frame_sizes()

    @property
    def seek_table_info(self):
        """A tuple: (frames_number, compressed_size, decompressed_size)
        1, Frames_number and compressed_size don't count the seek table
           frame (a zstd skippable frame at the end of the file).
        2, In write modes, the part of data that has not been flushed to
           frames is not counted.
        3, If the SeekableZstdFile object is closed, it's None.
        """
        if self._mode == _MODE_WRITE:
            return self._seek_table.get_info()
        elif self._mode == _MODE_READ:
            return self._buffer.raw.get_seek_table_info()
        else:
            return None

    @staticmethod
    def is_seekable_format_file(filename):
        """Check if a file is Zstandard Seekable Format file or 0-size file.

        It parses the seek table at the end of the file, returns True if no
        format error.

        filename can be either a file path (str/bytes/PathLike), or can be an
        existing file object in reading mode.
        """
        if isinstance(filename, (str, bytes, PathLike)):
            fp = io.open(filename, 'rb')
            is_file_path = True
        elif hasattr(filename, 'readable') and filename.readable() and hasattr(filename, 'seekable') and filename.seekable():
            fp = filename
            is_file_path = False
            orig_pos = fp.tell()
        else:
            raise TypeError('filename argument should be a str/bytes/PathLike object, or a file object that is readable and seekable.')
        table = SeekTable(read_mode=False)
        try:
            table.load_seek_table(fp, seek_to_0=False)
        except SeekableFormatError:
            ret = False
        else:
            ret = True
        finally:
            if is_file_path:
                fp.close()
            else:
                fp.seek(orig_pos)
        return ret