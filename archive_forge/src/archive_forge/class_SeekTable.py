from array import array
from bisect import bisect_right
from os.path import isfile
from struct import Struct
from warnings import warn
from pyzstd.zstdfile import ZstdDecompressReader, ZstdFile, \
class SeekTable:
    _s_2uint32 = Struct('<II')
    _s_3uint32 = Struct('<III')
    _s_footer = Struct('<IBI')

    def __init__(self, read_mode):
        self._read_mode = read_mode
        self._clear_seek_table()

    def _clear_seek_table(self):
        self._has_checksum = False
        self._seek_frame_size = 0
        self._file_size = 0
        self._frames_count = 0
        self._full_c_size = 0
        self._full_d_size = 0
        if self._read_mode:
            self._cumulated_c_size = array('q', [0])
            self._cumulated_d_size = array('q', [0])
        else:
            self._frames = array('I')

    def append_entry(self, compressed_size, decompressed_size):
        if compressed_size == 0:
            if decompressed_size == 0:
                return
            else:
                raise ValueError
        self._frames_count += 1
        self._full_c_size += compressed_size
        self._full_d_size += decompressed_size
        if self._read_mode:
            self._cumulated_c_size.append(self._full_c_size)
            self._cumulated_d_size.append(self._full_d_size)
        else:
            self._frames.append(compressed_size)
            self._frames.append(decompressed_size)

    def load_seek_table(self, fp, seek_to_0):
        fsize = fp.seek(0, 2)
        if fsize == 0:
            return
        elif fsize < 17:
            msg = 'File size is less than the minimal size (17 bytes) of Zstandard Seekable Format.'
            raise SeekableFormatError(msg)
        fp.seek(-9, 2)
        footer = fp.read(9)
        frames_number, descriptor, magic_number = self._s_footer.unpack(footer)
        if magic_number != 2408770225:
            msg = 'The last 4 bytes of the file is not Zstandard Seekable Format Magic Number (b"\\xb1\\xea\\x92\\x8f)". SeekableZstdFile class only supports Zstandard Seekable Format file or 0-size file. To read a zstd file that is not in Zstandard Seekable Format, use ZstdFile class.'
            raise SeekableFormatError(msg)
        self._has_checksum = descriptor & 128
        if descriptor & 124:
            msg = 'In Zstandard Seekable Format version %s, the Reserved_Bits in Seek_Table_Descriptor must be 0.' % __format_version__
            raise SeekableFormatError(msg)
        entry_size = 12 if self._has_checksum else 8
        skippable_frame_size = 17 + frames_number * entry_size
        if fsize < skippable_frame_size:
            raise SeekableFormatError('File size is less than expected size of the seek table frame.')
        fp.seek(-skippable_frame_size, 2)
        skippable_frame = fp.read(skippable_frame_size)
        skippable_magic_number, content_size = self._s_2uint32.unpack_from(skippable_frame, 0)
        if skippable_magic_number != 407710302:
            msg = "Seek table frame's Magic_Number is wrong."
            raise SeekableFormatError(msg)
        if content_size != skippable_frame_size - 8:
            msg = "Seek table frame's Frame_Size is wrong."
            raise SeekableFormatError(msg)
        if seek_to_0:
            fp.seek(0)
        offset = 8
        for idx in range(frames_number):
            if self._has_checksum:
                compressed_size, decompressed_size, checksum = self._s_3uint32.unpack_from(skippable_frame, offset)
                offset += 12
            else:
                compressed_size, decompressed_size = self._s_2uint32.unpack_from(skippable_frame, offset)
                offset += 8
            if compressed_size == 0 and decompressed_size != 0:
                msg = 'Wrong seek table. The index %d frame (0-based) is 0 size, but decompressed size is non-zero, this is impossible.' % idx
                raise SeekableFormatError(msg)
            self.append_entry(compressed_size, decompressed_size)
            if self._full_c_size > fsize - skippable_frame_size:
                msg = 'Wrong seek table. Since index %d frame (0-based), the cumulated compressed size is greater than file size.' % idx
                raise SeekableFormatError(msg)
        if self._full_c_size != fsize - skippable_frame_size:
            raise SeekableFormatError('The cumulated compressed size is wrong')
        self._seek_frame_size = skippable_frame_size
        self._file_size = fsize

    def index_by_dpos(self, pos):
        if pos < 0:
            pos = 0
        i = bisect_right(self._cumulated_d_size, pos)
        if i != self._frames_count + 1:
            return i
        else:
            return None

    def get_frame_sizes(self, i):
        return (self._cumulated_c_size[i - 1], self._cumulated_d_size[i - 1])

    def get_full_c_size(self):
        return self._full_c_size

    def get_full_d_size(self):
        return self._full_d_size

    def _merge_frames(self, max_frames):
        if self._frames_count <= max_frames:
            return
        arr = self._frames
        a, b = divmod(self._frames_count, max_frames)
        self._clear_seek_table()
        pos = 0
        for i in range(max_frames):
            length = (a + (1 if i < b else 0)) * 2
            c_size = 0
            d_size = 0
            for j in range(pos, pos + length, 2):
                c_size += arr[j]
                d_size += arr[j + 1]
            self.append_entry(c_size, d_size)
            pos += length

    def write_seek_table(self, fp):
        if self._frames_count > 4294967295:
            warn("SeekableZstdFile's seek table has %d entries, which exceeds the maximal value allowed by Zstandard Seekable Format (0xFFFFFFFF). The entries will be merged into 0xFFFFFFFF entries, this may reduce seeking performance." % self._frames_count, RuntimeWarning, 3)
            self._merge_frames(4294967295)
        offset = 0
        size = 17 + 8 * self._frames_count
        ba = bytearray(size)
        self._s_2uint32.pack_into(ba, offset, 407710302, size - 8)
        offset += 8
        for i in range(0, len(self._frames), 2):
            self._s_2uint32.pack_into(ba, offset, self._frames[i], self._frames[i + 1])
            offset += 8
        self._s_footer.pack_into(ba, offset, self._frames_count, 0, 2408770225)
        fp.write(ba)

    @property
    def seek_frame_size(self):
        return self._seek_frame_size

    @property
    def file_size(self):
        return self._file_size

    def __len__(self):
        return self._frames_count

    def get_info(self):
        return (self._frames_count, self._full_c_size, self._full_d_size)