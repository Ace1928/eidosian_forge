import _frozen_importlib_external as _bootstrap_external
from _frozen_importlib_external import _unpack_uint16, _unpack_uint32
import _frozen_importlib as _bootstrap  # for _verbose_message
import _imp  # for check_hash_based_pycs
import _io  # for open
import marshal  # for loads
import sys  # for modules
import time  # for mktime
import _warnings  # For warn()
def _read_directory(archive):
    try:
        fp = _io.open_code(archive)
    except OSError:
        raise ZipImportError(f"can't open Zip file: {archive!r}", path=archive)
    with fp:
        start_offset = fp.tell()
        try:
            try:
                fp.seek(-END_CENTRAL_DIR_SIZE, 2)
                header_position = fp.tell()
                buffer = fp.read(END_CENTRAL_DIR_SIZE)
            except OSError:
                raise ZipImportError(f"can't read Zip file: {archive!r}", path=archive)
            if len(buffer) != END_CENTRAL_DIR_SIZE:
                raise ZipImportError(f"can't read Zip file: {archive!r}", path=archive)
            if buffer[:4] != STRING_END_ARCHIVE:
                try:
                    fp.seek(0, 2)
                    file_size = fp.tell()
                except OSError:
                    raise ZipImportError(f"can't read Zip file: {archive!r}", path=archive)
                max_comment_start = max(file_size - MAX_COMMENT_LEN - END_CENTRAL_DIR_SIZE, 0)
                try:
                    fp.seek(max_comment_start)
                    data = fp.read()
                except OSError:
                    raise ZipImportError(f"can't read Zip file: {archive!r}", path=archive)
                pos = data.rfind(STRING_END_ARCHIVE)
                if pos < 0:
                    raise ZipImportError(f'not a Zip file: {archive!r}', path=archive)
                buffer = data[pos:pos + END_CENTRAL_DIR_SIZE]
                if len(buffer) != END_CENTRAL_DIR_SIZE:
                    raise ZipImportError(f'corrupt Zip file: {archive!r}', path=archive)
                header_position = file_size - len(data) + pos
            header_size = _unpack_uint32(buffer[12:16])
            header_offset = _unpack_uint32(buffer[16:20])
            if header_position < header_size:
                raise ZipImportError(f'bad central directory size: {archive!r}', path=archive)
            if header_position < header_offset:
                raise ZipImportError(f'bad central directory offset: {archive!r}', path=archive)
            header_position -= header_size
            arc_offset = header_position - header_offset
            if arc_offset < 0:
                raise ZipImportError(f'bad central directory size or offset: {archive!r}', path=archive)
            files = {}
            count = 0
            try:
                fp.seek(header_position)
            except OSError:
                raise ZipImportError(f"can't read Zip file: {archive!r}", path=archive)
            while True:
                buffer = fp.read(46)
                if len(buffer) < 4:
                    raise EOFError('EOF read where not expected')
                if buffer[:4] != b'PK\x01\x02':
                    break
                if len(buffer) != 46:
                    raise EOFError('EOF read where not expected')
                flags = _unpack_uint16(buffer[8:10])
                compress = _unpack_uint16(buffer[10:12])
                time = _unpack_uint16(buffer[12:14])
                date = _unpack_uint16(buffer[14:16])
                crc = _unpack_uint32(buffer[16:20])
                data_size = _unpack_uint32(buffer[20:24])
                file_size = _unpack_uint32(buffer[24:28])
                name_size = _unpack_uint16(buffer[28:30])
                extra_size = _unpack_uint16(buffer[30:32])
                comment_size = _unpack_uint16(buffer[32:34])
                file_offset = _unpack_uint32(buffer[42:46])
                header_size = name_size + extra_size + comment_size
                if file_offset > header_offset:
                    raise ZipImportError(f'bad local header offset: {archive!r}', path=archive)
                file_offset += arc_offset
                try:
                    name = fp.read(name_size)
                except OSError:
                    raise ZipImportError(f"can't read Zip file: {archive!r}", path=archive)
                if len(name) != name_size:
                    raise ZipImportError(f"can't read Zip file: {archive!r}", path=archive)
                try:
                    if len(fp.read(header_size - name_size)) != header_size - name_size:
                        raise ZipImportError(f"can't read Zip file: {archive!r}", path=archive)
                except OSError:
                    raise ZipImportError(f"can't read Zip file: {archive!r}", path=archive)
                if flags & 2048:
                    name = name.decode()
                else:
                    try:
                        name = name.decode('ascii')
                    except UnicodeDecodeError:
                        name = name.decode('latin1').translate(cp437_table)
                name = name.replace('/', path_sep)
                path = _bootstrap_external._path_join(archive, name)
                t = (path, compress, data_size, file_size, file_offset, time, date, crc)
                files[name] = t
                count += 1
        finally:
            fp.seek(start_offset)
    _bootstrap._verbose_message('zipimport: found {} names in {!r}', count, archive)
    return files