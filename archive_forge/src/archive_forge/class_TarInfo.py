from builtins import open as bltn_open
import sys
import os
import io
import shutil
import stat
import time
import struct
import copy
import re
import warnings
class TarInfo(object):
    """Informational class which holds the details about an
       archive member given by a tar header block.
       TarInfo objects are returned by TarFile.getmember(),
       TarFile.getmembers() and TarFile.gettarinfo() and are
       usually created internally.
    """
    __slots__ = dict(name='Name of the archive member.', mode='Permission bits.', uid='User ID of the user who originally stored this member.', gid='Group ID of the user who originally stored this member.', size='Size in bytes.', mtime='Time of last modification.', chksum='Header checksum.', type='File type. type is usually one of these constants: REGTYPE, AREGTYPE, LNKTYPE, SYMTYPE, DIRTYPE, FIFOTYPE, CONTTYPE, CHRTYPE, BLKTYPE, GNUTYPE_SPARSE.', linkname='Name of the target file name, which is only present in TarInfo objects of type LNKTYPE and SYMTYPE.', uname='User name.', gname='Group name.', devmajor='Device major number.', devminor='Device minor number.', offset='The tar header starts here.', offset_data="The file's data starts here.", pax_headers='A dictionary containing key-value pairs of an associated pax extended header.', sparse='Sparse member information.', tarfile=None, _sparse_structs=None, _link_target=None)

    def __init__(self, name=''):
        """Construct a TarInfo object. name is the optional name
           of the member.
        """
        self.name = name
        self.mode = 420
        self.uid = 0
        self.gid = 0
        self.size = 0
        self.mtime = 0
        self.chksum = 0
        self.type = REGTYPE
        self.linkname = ''
        self.uname = ''
        self.gname = ''
        self.devmajor = 0
        self.devminor = 0
        self.offset = 0
        self.offset_data = 0
        self.sparse = None
        self.pax_headers = {}

    @property
    def path(self):
        """In pax headers, "name" is called "path"."""
        return self.name

    @path.setter
    def path(self, name):
        self.name = name

    @property
    def linkpath(self):
        """In pax headers, "linkname" is called "linkpath"."""
        return self.linkname

    @linkpath.setter
    def linkpath(self, linkname):
        self.linkname = linkname

    def __repr__(self):
        return '<%s %r at %#x>' % (self.__class__.__name__, self.name, id(self))

    def replace(self, *, name=_KEEP, mtime=_KEEP, mode=_KEEP, linkname=_KEEP, uid=_KEEP, gid=_KEEP, uname=_KEEP, gname=_KEEP, deep=True, _KEEP=_KEEP):
        """Return a deep copy of self with the given attributes replaced.
        """
        if deep:
            result = copy.deepcopy(self)
        else:
            result = copy.copy(self)
        if name is not _KEEP:
            result.name = name
        if mtime is not _KEEP:
            result.mtime = mtime
        if mode is not _KEEP:
            result.mode = mode
        if linkname is not _KEEP:
            result.linkname = linkname
        if uid is not _KEEP:
            result.uid = uid
        if gid is not _KEEP:
            result.gid = gid
        if uname is not _KEEP:
            result.uname = uname
        if gname is not _KEEP:
            result.gname = gname
        return result

    def get_info(self):
        """Return the TarInfo's attributes as a dictionary.
        """
        if self.mode is None:
            mode = None
        else:
            mode = self.mode & 4095
        info = {'name': self.name, 'mode': mode, 'uid': self.uid, 'gid': self.gid, 'size': self.size, 'mtime': self.mtime, 'chksum': self.chksum, 'type': self.type, 'linkname': self.linkname, 'uname': self.uname, 'gname': self.gname, 'devmajor': self.devmajor, 'devminor': self.devminor}
        if info['type'] == DIRTYPE and (not info['name'].endswith('/')):
            info['name'] += '/'
        return info

    def tobuf(self, format=DEFAULT_FORMAT, encoding=ENCODING, errors='surrogateescape'):
        """Return a tar header as a string of 512 byte blocks.
        """
        info = self.get_info()
        for name, value in info.items():
            if value is None:
                raise ValueError('%s may not be None' % name)
        if format == USTAR_FORMAT:
            return self.create_ustar_header(info, encoding, errors)
        elif format == GNU_FORMAT:
            return self.create_gnu_header(info, encoding, errors)
        elif format == PAX_FORMAT:
            return self.create_pax_header(info, encoding)
        else:
            raise ValueError('invalid format')

    def create_ustar_header(self, info, encoding, errors):
        """Return the object as a ustar header block.
        """
        info['magic'] = POSIX_MAGIC
        if len(info['linkname'].encode(encoding, errors)) > LENGTH_LINK:
            raise ValueError('linkname is too long')
        if len(info['name'].encode(encoding, errors)) > LENGTH_NAME:
            info['prefix'], info['name'] = self._posix_split_name(info['name'], encoding, errors)
        return self._create_header(info, USTAR_FORMAT, encoding, errors)

    def create_gnu_header(self, info, encoding, errors):
        """Return the object as a GNU header block sequence.
        """
        info['magic'] = GNU_MAGIC
        buf = b''
        if len(info['linkname'].encode(encoding, errors)) > LENGTH_LINK:
            buf += self._create_gnu_long_header(info['linkname'], GNUTYPE_LONGLINK, encoding, errors)
        if len(info['name'].encode(encoding, errors)) > LENGTH_NAME:
            buf += self._create_gnu_long_header(info['name'], GNUTYPE_LONGNAME, encoding, errors)
        return buf + self._create_header(info, GNU_FORMAT, encoding, errors)

    def create_pax_header(self, info, encoding):
        """Return the object as a ustar header block. If it cannot be
           represented this way, prepend a pax extended header sequence
           with supplement information.
        """
        info['magic'] = POSIX_MAGIC
        pax_headers = self.pax_headers.copy()
        for name, hname, length in (('name', 'path', LENGTH_NAME), ('linkname', 'linkpath', LENGTH_LINK), ('uname', 'uname', 32), ('gname', 'gname', 32)):
            if hname in pax_headers:
                continue
            try:
                info[name].encode('ascii', 'strict')
            except UnicodeEncodeError:
                pax_headers[hname] = info[name]
                continue
            if len(info[name]) > length:
                pax_headers[hname] = info[name]
        for name, digits in (('uid', 8), ('gid', 8), ('size', 12), ('mtime', 12)):
            needs_pax = False
            val = info[name]
            val_is_float = isinstance(val, float)
            val_int = round(val) if val_is_float else val
            if not 0 <= val_int < 8 ** (digits - 1):
                info[name] = 0
                needs_pax = True
            elif val_is_float:
                info[name] = val_int
                needs_pax = True
            if needs_pax and name not in pax_headers:
                pax_headers[name] = str(val)
        if pax_headers:
            buf = self._create_pax_generic_header(pax_headers, XHDTYPE, encoding)
        else:
            buf = b''
        return buf + self._create_header(info, USTAR_FORMAT, 'ascii', 'replace')

    @classmethod
    def create_pax_global_header(cls, pax_headers):
        """Return the object as a pax global header block sequence.
        """
        return cls._create_pax_generic_header(pax_headers, XGLTYPE, 'utf-8')

    def _posix_split_name(self, name, encoding, errors):
        """Split a name longer than 100 chars into a prefix
           and a name part.
        """
        components = name.split('/')
        for i in range(1, len(components)):
            prefix = '/'.join(components[:i])
            name = '/'.join(components[i:])
            if len(prefix.encode(encoding, errors)) <= LENGTH_PREFIX and len(name.encode(encoding, errors)) <= LENGTH_NAME:
                break
        else:
            raise ValueError('name is too long')
        return (prefix, name)

    @staticmethod
    def _create_header(info, format, encoding, errors):
        """Return a header block. info is a dictionary with file
           information, format must be one of the *_FORMAT constants.
        """
        has_device_fields = info.get('type') in (CHRTYPE, BLKTYPE)
        if has_device_fields:
            devmajor = itn(info.get('devmajor', 0), 8, format)
            devminor = itn(info.get('devminor', 0), 8, format)
        else:
            devmajor = stn('', 8, encoding, errors)
            devminor = stn('', 8, encoding, errors)
        filetype = info.get('type', REGTYPE)
        if filetype is None:
            raise ValueError('TarInfo.type must not be None')
        parts = [stn(info.get('name', ''), 100, encoding, errors), itn(info.get('mode', 0) & 4095, 8, format), itn(info.get('uid', 0), 8, format), itn(info.get('gid', 0), 8, format), itn(info.get('size', 0), 12, format), itn(info.get('mtime', 0), 12, format), b'        ', filetype, stn(info.get('linkname', ''), 100, encoding, errors), info.get('magic', POSIX_MAGIC), stn(info.get('uname', ''), 32, encoding, errors), stn(info.get('gname', ''), 32, encoding, errors), devmajor, devminor, stn(info.get('prefix', ''), 155, encoding, errors)]
        buf = struct.pack('%ds' % BLOCKSIZE, b''.join(parts))
        chksum = calc_chksums(buf[-BLOCKSIZE:])[0]
        buf = buf[:-364] + bytes('%06o\x00' % chksum, 'ascii') + buf[-357:]
        return buf

    @staticmethod
    def _create_payload(payload):
        """Return the string payload filled with zero bytes
           up to the next 512 byte border.
        """
        blocks, remainder = divmod(len(payload), BLOCKSIZE)
        if remainder > 0:
            payload += (BLOCKSIZE - remainder) * NUL
        return payload

    @classmethod
    def _create_gnu_long_header(cls, name, type, encoding, errors):
        """Return a GNUTYPE_LONGNAME or GNUTYPE_LONGLINK sequence
           for name.
        """
        name = name.encode(encoding, errors) + NUL
        info = {}
        info['name'] = '././@LongLink'
        info['type'] = type
        info['size'] = len(name)
        info['magic'] = GNU_MAGIC
        return cls._create_header(info, USTAR_FORMAT, encoding, errors) + cls._create_payload(name)

    @classmethod
    def _create_pax_generic_header(cls, pax_headers, type, encoding):
        """Return a POSIX.1-2008 extended or global header sequence
           that contains a list of keyword, value pairs. The values
           must be strings.
        """
        binary = False
        for keyword, value in pax_headers.items():
            try:
                value.encode('utf-8', 'strict')
            except UnicodeEncodeError:
                binary = True
                break
        records = b''
        if binary:
            records += b'21 hdrcharset=BINARY\n'
        for keyword, value in pax_headers.items():
            keyword = keyword.encode('utf-8')
            if binary:
                value = value.encode(encoding, 'surrogateescape')
            else:
                value = value.encode('utf-8')
            l = len(keyword) + len(value) + 3
            n = p = 0
            while True:
                n = l + len(str(p))
                if n == p:
                    break
                p = n
            records += bytes(str(p), 'ascii') + b' ' + keyword + b'=' + value + b'\n'
        info = {}
        info['name'] = '././@PaxHeader'
        info['type'] = type
        info['size'] = len(records)
        info['magic'] = POSIX_MAGIC
        return cls._create_header(info, USTAR_FORMAT, 'ascii', 'replace') + cls._create_payload(records)

    @classmethod
    def frombuf(cls, buf, encoding, errors):
        """Construct a TarInfo object from a 512 byte bytes object.
        """
        if len(buf) == 0:
            raise EmptyHeaderError('empty header')
        if len(buf) != BLOCKSIZE:
            raise TruncatedHeaderError('truncated header')
        if buf.count(NUL) == BLOCKSIZE:
            raise EOFHeaderError('end of file header')
        chksum = nti(buf[148:156])
        if chksum not in calc_chksums(buf):
            raise InvalidHeaderError('bad checksum')
        obj = cls()
        obj.name = nts(buf[0:100], encoding, errors)
        obj.mode = nti(buf[100:108])
        obj.uid = nti(buf[108:116])
        obj.gid = nti(buf[116:124])
        obj.size = nti(buf[124:136])
        obj.mtime = nti(buf[136:148])
        obj.chksum = chksum
        obj.type = buf[156:157]
        obj.linkname = nts(buf[157:257], encoding, errors)
        obj.uname = nts(buf[265:297], encoding, errors)
        obj.gname = nts(buf[297:329], encoding, errors)
        obj.devmajor = nti(buf[329:337])
        obj.devminor = nti(buf[337:345])
        prefix = nts(buf[345:500], encoding, errors)
        if obj.type == AREGTYPE and obj.name.endswith('/'):
            obj.type = DIRTYPE
        if obj.type == GNUTYPE_SPARSE:
            pos = 386
            structs = []
            for i in range(4):
                try:
                    offset = nti(buf[pos:pos + 12])
                    numbytes = nti(buf[pos + 12:pos + 24])
                except ValueError:
                    break
                structs.append((offset, numbytes))
                pos += 24
            isextended = bool(buf[482])
            origsize = nti(buf[483:495])
            obj._sparse_structs = (structs, isextended, origsize)
        if obj.isdir():
            obj.name = obj.name.rstrip('/')
        if prefix and obj.type not in GNU_TYPES:
            obj.name = prefix + '/' + obj.name
        return obj

    @classmethod
    def fromtarfile(cls, tarfile):
        """Return the next TarInfo object from TarFile object
           tarfile.
        """
        buf = tarfile.fileobj.read(BLOCKSIZE)
        obj = cls.frombuf(buf, tarfile.encoding, tarfile.errors)
        obj.offset = tarfile.fileobj.tell() - BLOCKSIZE
        return obj._proc_member(tarfile)

    def _proc_member(self, tarfile):
        """Choose the right processing method depending on
           the type and call it.
        """
        if self.type in (GNUTYPE_LONGNAME, GNUTYPE_LONGLINK):
            return self._proc_gnulong(tarfile)
        elif self.type == GNUTYPE_SPARSE:
            return self._proc_sparse(tarfile)
        elif self.type in (XHDTYPE, XGLTYPE, SOLARIS_XHDTYPE):
            return self._proc_pax(tarfile)
        else:
            return self._proc_builtin(tarfile)

    def _proc_builtin(self, tarfile):
        """Process a builtin type or an unknown type which
           will be treated as a regular file.
        """
        self.offset_data = tarfile.fileobj.tell()
        offset = self.offset_data
        if self.isreg() or self.type not in SUPPORTED_TYPES:
            offset += self._block(self.size)
        tarfile.offset = offset
        self._apply_pax_info(tarfile.pax_headers, tarfile.encoding, tarfile.errors)
        if self.isdir():
            self.name = self.name.rstrip('/')
        return self

    def _proc_gnulong(self, tarfile):
        """Process the blocks that hold a GNU longname
           or longlink member.
        """
        buf = tarfile.fileobj.read(self._block(self.size))
        try:
            next = self.fromtarfile(tarfile)
        except HeaderError as e:
            raise SubsequentHeaderError(str(e)) from None
        next.offset = self.offset
        if self.type == GNUTYPE_LONGNAME:
            next.name = nts(buf, tarfile.encoding, tarfile.errors)
        elif self.type == GNUTYPE_LONGLINK:
            next.linkname = nts(buf, tarfile.encoding, tarfile.errors)
        if next.isdir():
            next.name = next.name.removesuffix('/')
        return next

    def _proc_sparse(self, tarfile):
        """Process a GNU sparse header plus extra headers.
        """
        structs, isextended, origsize = self._sparse_structs
        del self._sparse_structs
        while isextended:
            buf = tarfile.fileobj.read(BLOCKSIZE)
            pos = 0
            for i in range(21):
                try:
                    offset = nti(buf[pos:pos + 12])
                    numbytes = nti(buf[pos + 12:pos + 24])
                except ValueError:
                    break
                if offset and numbytes:
                    structs.append((offset, numbytes))
                pos += 24
            isextended = bool(buf[504])
        self.sparse = structs
        self.offset_data = tarfile.fileobj.tell()
        tarfile.offset = self.offset_data + self._block(self.size)
        self.size = origsize
        return self

    def _proc_pax(self, tarfile):
        """Process an extended or global header as described in
           POSIX.1-2008.
        """
        buf = tarfile.fileobj.read(self._block(self.size))
        if self.type == XGLTYPE:
            pax_headers = tarfile.pax_headers
        else:
            pax_headers = tarfile.pax_headers.copy()
        match = re.search(b'\\d+ hdrcharset=([^\\n]+)\\n', buf)
        if match is not None:
            pax_headers['hdrcharset'] = match.group(1).decode('utf-8')
        hdrcharset = pax_headers.get('hdrcharset')
        if hdrcharset == 'BINARY':
            encoding = tarfile.encoding
        else:
            encoding = 'utf-8'
        regex = re.compile(b'(\\d+) ([^=]+)=')
        pos = 0
        while True:
            match = regex.match(buf, pos)
            if not match:
                break
            length, keyword = match.groups()
            length = int(length)
            if length == 0:
                raise InvalidHeaderError('invalid header')
            value = buf[match.end(2) + 1:match.start(1) + length - 1]
            keyword = self._decode_pax_field(keyword, 'utf-8', 'utf-8', tarfile.errors)
            if keyword in PAX_NAME_FIELDS:
                value = self._decode_pax_field(value, encoding, tarfile.encoding, tarfile.errors)
            else:
                value = self._decode_pax_field(value, 'utf-8', 'utf-8', tarfile.errors)
            pax_headers[keyword] = value
            pos += length
        try:
            next = self.fromtarfile(tarfile)
        except HeaderError as e:
            raise SubsequentHeaderError(str(e)) from None
        if 'GNU.sparse.map' in pax_headers:
            self._proc_gnusparse_01(next, pax_headers)
        elif 'GNU.sparse.size' in pax_headers:
            self._proc_gnusparse_00(next, pax_headers, buf)
        elif pax_headers.get('GNU.sparse.major') == '1' and pax_headers.get('GNU.sparse.minor') == '0':
            self._proc_gnusparse_10(next, pax_headers, tarfile)
        if self.type in (XHDTYPE, SOLARIS_XHDTYPE):
            next._apply_pax_info(pax_headers, tarfile.encoding, tarfile.errors)
            next.offset = self.offset
            if 'size' in pax_headers:
                offset = next.offset_data
                if next.isreg() or next.type not in SUPPORTED_TYPES:
                    offset += next._block(next.size)
                tarfile.offset = offset
        return next

    def _proc_gnusparse_00(self, next, pax_headers, buf):
        """Process a GNU tar extended sparse header, version 0.0.
        """
        offsets = []
        for match in re.finditer(b'\\d+ GNU.sparse.offset=(\\d+)\\n', buf):
            offsets.append(int(match.group(1)))
        numbytes = []
        for match in re.finditer(b'\\d+ GNU.sparse.numbytes=(\\d+)\\n', buf):
            numbytes.append(int(match.group(1)))
        next.sparse = list(zip(offsets, numbytes))

    def _proc_gnusparse_01(self, next, pax_headers):
        """Process a GNU tar extended sparse header, version 0.1.
        """
        sparse = [int(x) for x in pax_headers['GNU.sparse.map'].split(',')]
        next.sparse = list(zip(sparse[::2], sparse[1::2]))

    def _proc_gnusparse_10(self, next, pax_headers, tarfile):
        """Process a GNU tar extended sparse header, version 1.0.
        """
        fields = None
        sparse = []
        buf = tarfile.fileobj.read(BLOCKSIZE)
        fields, buf = buf.split(b'\n', 1)
        fields = int(fields)
        while len(sparse) < fields * 2:
            if b'\n' not in buf:
                buf += tarfile.fileobj.read(BLOCKSIZE)
            number, buf = buf.split(b'\n', 1)
            sparse.append(int(number))
        next.offset_data = tarfile.fileobj.tell()
        next.sparse = list(zip(sparse[::2], sparse[1::2]))

    def _apply_pax_info(self, pax_headers, encoding, errors):
        """Replace fields with supplemental information from a previous
           pax extended or global header.
        """
        for keyword, value in pax_headers.items():
            if keyword == 'GNU.sparse.name':
                setattr(self, 'path', value)
            elif keyword == 'GNU.sparse.size':
                setattr(self, 'size', int(value))
            elif keyword == 'GNU.sparse.realsize':
                setattr(self, 'size', int(value))
            elif keyword in PAX_FIELDS:
                if keyword in PAX_NUMBER_FIELDS:
                    try:
                        value = PAX_NUMBER_FIELDS[keyword](value)
                    except ValueError:
                        value = 0
                if keyword == 'path':
                    value = value.rstrip('/')
                setattr(self, keyword, value)
        self.pax_headers = pax_headers.copy()

    def _decode_pax_field(self, value, encoding, fallback_encoding, fallback_errors):
        """Decode a single field from a pax record.
        """
        try:
            return value.decode(encoding, 'strict')
        except UnicodeDecodeError:
            return value.decode(fallback_encoding, fallback_errors)

    def _block(self, count):
        """Round up a byte count by BLOCKSIZE and return it,
           e.g. _block(834) => 1024.
        """
        blocks, remainder = divmod(count, BLOCKSIZE)
        if remainder:
            blocks += 1
        return blocks * BLOCKSIZE

    def isreg(self):
        """Return True if the Tarinfo object is a regular file."""
        return self.type in REGULAR_TYPES

    def isfile(self):
        """Return True if the Tarinfo object is a regular file."""
        return self.isreg()

    def isdir(self):
        """Return True if it is a directory."""
        return self.type == DIRTYPE

    def issym(self):
        """Return True if it is a symbolic link."""
        return self.type == SYMTYPE

    def islnk(self):
        """Return True if it is a hard link."""
        return self.type == LNKTYPE

    def ischr(self):
        """Return True if it is a character device."""
        return self.type == CHRTYPE

    def isblk(self):
        """Return True if it is a block device."""
        return self.type == BLKTYPE

    def isfifo(self):
        """Return True if it is a FIFO."""
        return self.type == FIFOTYPE

    def issparse(self):
        return self.sparse is not None

    def isdev(self):
        """Return True if it is one of character device, block device or FIFO."""
        return self.type in (CHRTYPE, BLKTYPE, FIFOTYPE)