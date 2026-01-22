from __future__ import division, print_function
import sys
import os
import io
import re
import glob
import math
import zlib
import time
import json
import enum
import struct
import pathlib
import warnings
import binascii
import tempfile
import datetime
import threading
import collections
import multiprocessing
import concurrent.futures
import numpy
class TiffPage(object):
    """TIFF image file directory (IFD).

    Attributes
    ----------
    index : int
        Index of page in file.
    dtype : numpy.dtype or None
        Data type (native byte order) of the image in IFD.
    shape : tuple
        Dimensions of the image in IFD.
    axes : str
        Axes label codes:
        'X' width, 'Y' height, 'S' sample, 'I' image series|page|plane,
        'Z' depth, 'C' color|em-wavelength|channel, 'E' ex-wavelength|lambda,
        'T' time, 'R' region|tile, 'A' angle, 'P' phase, 'H' lifetime,
        'L' exposure, 'V' event, 'Q' unknown, '_' missing
    tags : dict
        Dictionary of tags in IFD. {tag.name: TiffTag}
    colormap : numpy.ndarray
        Color look up table, if exists.

    All attributes are read-only.

    Notes
    -----
    The internal, normalized '_shape' attribute is 6 dimensional:

    0 : number planes/images  (stk, ij).
    1 : planar samplesperpixel.
    2 : imagedepth Z  (sgi).
    3 : imagelength Y.
    4 : imagewidth X.
    5 : contig samplesperpixel.

    """
    imagewidth = 0
    imagelength = 0
    imagedepth = 1
    tilewidth = 0
    tilelength = 0
    tiledepth = 1
    bitspersample = 1
    samplesperpixel = 1
    sampleformat = 1
    rowsperstrip = 2 ** 32 - 1
    compression = 1
    planarconfig = 1
    fillorder = 1
    photometric = 0
    predictor = 1
    extrasamples = 1
    colormap = None
    software = ''
    description = ''
    description1 = ''

    def __init__(self, parent, index, keyframe=None):
        """Initialize instance from file.

        The file handle position must be at offset to a valid IFD.

        """
        self.parent = parent
        self.index = index
        self.shape = ()
        self._shape = ()
        self.dtype = None
        self._dtype = None
        self.axes = ''
        self.tags = {}
        self.dataoffsets = ()
        self.databytecounts = ()
        fh = parent.filehandle
        self.offset = fh.tell()
        try:
            tagno = struct.unpack(parent.tagnoformat, fh.read(parent.tagnosize))[0]
            if tagno > 4096:
                raise ValueError('suspicious number of tags')
        except Exception:
            raise ValueError('corrupted tag list at offset %i' % self.offset)
        tagsize = parent.tagsize
        data = fh.read(tagsize * tagno)
        tags = self.tags
        index = -tagsize
        for _ in range(tagno):
            index += tagsize
            try:
                tag = TiffTag(self.parent, data[index:index + tagsize])
            except TiffTag.Error as e:
                warnings.warn(str(e))
                continue
            tagname = tag.name
            if tagname not in tags:
                name = tagname
                tags[name] = tag
            else:
                i = 1
                while True:
                    name = '%s%i' % (tagname, i)
                    if name not in tags:
                        tags[name] = tag
                        break
            name = TIFF.TAG_ATTRIBUTES.get(name, '')
            if name:
                if name[:3] in 'sof des' and (not isinstance(tag.value, str)):
                    pass
                else:
                    setattr(self, name, tag.value)
        if not tags:
            return
        if self.is_andor:
            self.andor_tags
        elif self.is_epics:
            self.epics_tags
        if self.is_lsm or (self.index and self.parent.is_lsm):
            self.tags['BitsPerSample']._fix_lsm_bitspersample(self)
        if self.is_vista or (self.index and self.parent.is_vista):
            self.imagedepth = 1
        if self.is_stk and 'UIC1tag' in tags and (not tags['UIC1tag'].value):
            uic1tag = tags['UIC1tag']
            fh.seek(uic1tag.valueoffset)
            tags['UIC1tag'].value = read_uic1tag(fh, self.parent.byteorder, uic1tag.dtype, uic1tag.count, None, tags['UIC2tag'].count)
        if 'IJMetadata' in tags:
            try:
                tags['IJMetadata'].value = imagej_metadata(tags['IJMetadata'].value, tags['IJMetadataByteCounts'].value, self.parent.byteorder)
            except Exception as e:
                warnings.warn(str(e))
        if 'BitsPerSample' in tags:
            tag = tags['BitsPerSample']
            if tag.count == 1:
                self.bitspersample = tag.value
            else:
                value = tag.value[:self.samplesperpixel]
                if any((v - value[0] for v in value)):
                    self.bitspersample = value
                else:
                    self.bitspersample = value[0]
        if 'SampleFormat' in tags:
            tag = tags['SampleFormat']
            if tag.count == 1:
                self.sampleformat = tag.value
            else:
                value = tag.value[:self.samplesperpixel]
                if any((v - value[0] for v in value)):
                    self.sampleformat = value
                else:
                    self.sampleformat = value[0]
        if 'ImageLength' in tags:
            if 'RowsPerStrip' not in tags or tags['RowsPerStrip'].count > 1:
                self.rowsperstrip = self.imagelength
        dtype = (self.sampleformat, self.bitspersample)
        dtype = TIFF.SAMPLE_DTYPES.get(dtype, None)
        if dtype is not None:
            dtype = numpy.dtype(dtype)
        self.dtype = self._dtype = dtype
        imagelength = self.imagelength
        imagewidth = self.imagewidth
        imagedepth = self.imagedepth
        samplesperpixel = self.samplesperpixel
        if self.is_stk:
            assert self.imagedepth == 1
            uictag = tags['UIC2tag'].value
            planes = tags['UIC2tag'].count
            if self.planarconfig == 1:
                self._shape = (planes, 1, 1, imagelength, imagewidth, samplesperpixel)
                if samplesperpixel == 1:
                    self.shape = (planes, imagelength, imagewidth)
                    self.axes = 'YX'
                else:
                    self.shape = (planes, imagelength, imagewidth, samplesperpixel)
                    self.axes = 'YXS'
            else:
                self._shape = (planes, samplesperpixel, 1, imagelength, imagewidth, 1)
                if samplesperpixel == 1:
                    self.shape = (planes, imagelength, imagewidth)
                    self.axes = 'YX'
                else:
                    self.shape = (planes, samplesperpixel, imagelength, imagewidth)
                    self.axes = 'SYX'
            if planes == 1:
                self.shape = self.shape[1:]
            elif numpy.all(uictag['ZDistance'] != 0):
                self.axes = 'Z' + self.axes
            elif numpy.all(numpy.diff(uictag['TimeCreated']) != 0):
                self.axes = 'T' + self.axes
            else:
                self.axes = 'I' + self.axes
        elif self.photometric == 2 or samplesperpixel > 1:
            if self.planarconfig == 1:
                self._shape = (1, 1, imagedepth, imagelength, imagewidth, samplesperpixel)
                if imagedepth == 1:
                    self.shape = (imagelength, imagewidth, samplesperpixel)
                    self.axes = 'YXS'
                else:
                    self.shape = (imagedepth, imagelength, imagewidth, samplesperpixel)
                    self.axes = 'ZYXS'
            else:
                self._shape = (1, samplesperpixel, imagedepth, imagelength, imagewidth, 1)
                if imagedepth == 1:
                    self.shape = (samplesperpixel, imagelength, imagewidth)
                    self.axes = 'SYX'
                else:
                    self.shape = (samplesperpixel, imagedepth, imagelength, imagewidth)
                    self.axes = 'SZYX'
        else:
            self._shape = (1, 1, imagedepth, imagelength, imagewidth, 1)
            if imagedepth == 1:
                self.shape = (imagelength, imagewidth)
                self.axes = 'YX'
            else:
                self.shape = (imagedepth, imagelength, imagewidth)
                self.axes = 'ZYX'
        if 'TileOffsets' in tags:
            self.dataoffsets = tags['TileOffsets'].value
        elif 'StripOffsets' in tags:
            self.dataoffsets = tags['StripOffsets'].value
        else:
            self.dataoffsets = (0,)
        if 'TileByteCounts' in tags:
            self.databytecounts = tags['TileByteCounts'].value
        elif 'StripByteCounts' in tags:
            self.databytecounts = tags['StripByteCounts'].value
        else:
            self.databytecounts = (product(self.shape) * (self.bitspersample // 8),)
            if self.compression != 1:
                warnings.warn('required ByteCounts tag is missing')
        assert len(self.shape) == len(self.axes)

    def asarray(self, out=None, squeeze=True, lock=None, reopen=True, maxsize=2 ** 44, validate=True):
        """Read image data from file and return as numpy array.

        Raise ValueError if format is unsupported.

        Parameters
        ----------
        out : numpy.ndarray, str, or file-like object; optional
            Buffer where image data will be saved.
            If None (default), a new array will be created.
            If numpy.ndarray, a writable array of compatible dtype and shape.
            If 'memmap', directly memory-map the image data in the TIFF file
            if possible; else create a memory-mapped array in a temporary file.
            If str or open file, the file name or file object used to
            create a memory-map to an array stored in a binary file on disk.
        squeeze : bool
            If True, all length-1 dimensions (except X and Y) are
            squeezed out from the array.
            If False, the shape of the returned array might be different from
            the page.shape.
        lock : {RLock, NullContext}
            A reentrant lock used to synchronize reads from file.
            If None (default), the lock of the parent's filehandle is used.
        reopen : bool
            If True (default) and the parent file handle is closed, the file
            is temporarily re-opened and closed if no exception occurs.
        maxsize: int or None
            Maximum size of data before a ValueError is raised.
            Can be used to catch DOS. Default: 16 TB.
        validate : bool
            If True (default), validate various parameters.
            If None, only validate parameters and return None.

        """
        self_ = self
        self = self.keyframe
        if not self._shape or product(self._shape) == 0:
            return
        tags = self.tags
        if validate or validate is None:
            if maxsize and product(self._shape) > maxsize:
                raise ValueError('data are too large %s' % str(self._shape))
            if self.dtype is None:
                raise ValueError('data type not supported: %s%i' % (self.sampleformat, self.bitspersample))
            if self.compression not in TIFF.DECOMPESSORS:
                raise ValueError('cannot decompress %s' % self.compression.name)
            if 'SampleFormat' in tags:
                tag = tags['SampleFormat']
                if tag.count != 1 and any((i - tag.value[0] for i in tag.value)):
                    raise ValueError('sample formats do not match %s' % tag.value)
            if self.is_chroma_subsampled and (self.compression != 7 or self.planarconfig == 2):
                raise NotImplementedError('chroma subsampling not supported')
            if validate is None:
                return
        fh = self_.parent.filehandle
        lock = fh.lock if lock is None else lock
        with lock:
            closed = fh.closed
            if closed:
                if reopen:
                    fh.open()
                else:
                    raise IOError('file handle is closed')
        dtype = self._dtype
        shape = self._shape
        imagewidth = self.imagewidth
        imagelength = self.imagelength
        imagedepth = self.imagedepth
        bitspersample = self.bitspersample
        typecode = self.parent.byteorder + dtype.char
        lsb2msb = self.fillorder == 2
        offsets, bytecounts = self_.offsets_bytecounts
        istiled = self.is_tiled
        if istiled:
            tilewidth = self.tilewidth
            tilelength = self.tilelength
            tiledepth = self.tiledepth
            tw = (imagewidth + tilewidth - 1) // tilewidth
            tl = (imagelength + tilelength - 1) // tilelength
            td = (imagedepth + tiledepth - 1) // tiledepth
            shape = (shape[0], shape[1], td * tiledepth, tl * tilelength, tw * tilewidth, shape[-1])
            tileshape = (tiledepth, tilelength, tilewidth, shape[-1])
            runlen = tilewidth
        else:
            runlen = imagewidth
        if self.planarconfig == 1:
            runlen *= self.samplesperpixel
        if out == 'memmap' and self.is_memmappable:
            with lock:
                result = fh.memmap_array(typecode, shape, offset=offsets[0])
        elif self.is_contiguous:
            if out is not None:
                out = create_output(out, shape, dtype)
            with lock:
                fh.seek(offsets[0])
                result = fh.read_array(typecode, product(shape), out=out)
            if out is None and (not result.dtype.isnative):
                result.byteswap(True)
                result = result.newbyteorder()
            if lsb2msb:
                reverse_bitorder(result)
        else:
            result = create_output(out, shape, dtype)
            decompress = TIFF.DECOMPESSORS[self.compression]
            if self.compression == 7:
                if bitspersample not in (8, 12):
                    raise ValueError('unsupported JPEG precision %i' % bitspersample)
                if 'JPEGTables' in tags:
                    table = tags['JPEGTables'].value
                else:
                    table = b''
                unpack = identityfunc
                colorspace = TIFF.PHOTOMETRIC(self.photometric).name

                def decompress(x, func=decompress, table=table, bitspersample=bitspersample, colorspace=colorspace):
                    return func(x, table, bitspersample, colorspace).reshape(-1)
            elif bitspersample in (8, 16, 32, 64, 128):
                if bitspersample * runlen % 8:
                    raise ValueError('data and sample size mismatch')

                def unpack(x, typecode=typecode):
                    if self.predictor == 3:
                        typecode = dtype.char
                    try:
                        return numpy.frombuffer(x, typecode)
                    except ValueError:
                        xlen = len(x) // (bitspersample // 8) * (bitspersample // 8)
                        return numpy.frombuffer(x[:xlen], typecode)
            elif isinstance(bitspersample, tuple):

                def unpack(x, typecode=typecode, bitspersample=bitspersample):
                    return unpack_rgb(x, typecode, bitspersample)
            else:

                def unpack(x, typecode=typecode, bitspersample=bitspersample, runlen=runlen):
                    return unpack_ints(x, typecode, bitspersample, runlen)
            if istiled:
                writable = None
                tw, tl, td, pl = (0, 0, 0, 0)
                for tile in buffered_read(fh, lock, offsets, bytecounts):
                    if lsb2msb:
                        tile = reverse_bitorder(tile)
                    tile = decompress(tile)
                    tile = unpack(tile)
                    try:
                        tile.shape = tileshape
                    except ValueError:
                        warnings.warn('invalid tile data')
                        t = numpy.zeros(tileshape, dtype).reshape(-1)
                        s = min(tile.size, t.size)
                        t[:s] = tile[:s]
                        tile = t.reshape(tileshape)
                    if self.predictor == 2:
                        if writable is None:
                            writable = tile.flags['WRITEABLE']
                        if writable:
                            numpy.cumsum(tile, axis=-2, dtype=dtype, out=tile)
                        else:
                            tile = numpy.cumsum(tile, axis=-2, dtype=dtype)
                    elif self.predictor == 3:
                        raise NotImplementedError()
                    result[0, pl, td:td + tiledepth, tl:tl + tilelength, tw:tw + tilewidth, :] = tile
                    del tile
                    tw += tilewidth
                    if tw >= shape[4]:
                        tw, tl = (0, tl + tilelength)
                        if tl >= shape[3]:
                            tl, td = (0, td + tiledepth)
                            if td >= shape[2]:
                                td, pl = (0, pl + 1)
                result = result[..., :imagedepth, :imagelength, :imagewidth, :]
            else:
                strip_size = self.rowsperstrip * self.imagewidth
                if self.planarconfig == 1:
                    strip_size *= self.samplesperpixel
                result = result.reshape(-1)
                index = 0
                for strip in buffered_read(fh, lock, offsets, bytecounts):
                    if lsb2msb:
                        strip = reverse_bitorder(strip)
                    strip = decompress(strip)
                    strip = unpack(strip)
                    size = min(result.size, strip.size, strip_size, result.size - index)
                    result[index:index + size] = strip[:size]
                    del strip
                    index += size
        result.shape = self._shape
        if self.predictor != 1 and (not (istiled and (not self.is_contiguous))):
            if self.parent.is_lsm and self.compression == 1:
                pass
            elif self.predictor == 2:
                numpy.cumsum(result, axis=-2, dtype=dtype, out=result)
            elif self.predictor == 3:
                result = decode_floats(result)
        if squeeze:
            try:
                result.shape = self.shape
            except ValueError:
                warnings.warn('failed to reshape from %s to %s' % (str(result.shape), str(self.shape)))
        if closed:
            fh.close()
        return result

    def asrgb(self, uint8=False, alpha=None, colormap=None, dmin=None, dmax=None, *args, **kwargs):
        """Return image data as RGB(A).

        Work in progress.

        """
        data = self.asarray(*args, **kwargs)
        self = self.keyframe
        photometric = self.photometric
        PHOTOMETRIC = TIFF.PHOTOMETRIC
        if photometric == PHOTOMETRIC.PALETTE:
            colormap = self.colormap
            if colormap.shape[1] < 2 ** self.bitspersample or self.dtype.char not in 'BH':
                raise ValueError('cannot apply colormap')
            if uint8:
                if colormap.max() > 255:
                    colormap >>= 8
                colormap = colormap.astype('uint8')
            if 'S' in self.axes:
                data = data[..., 0] if self.planarconfig == 1 else data[0]
            data = apply_colormap(data, colormap)
        elif photometric == PHOTOMETRIC.RGB:
            if 'ExtraSamples' in self.tags:
                if alpha is None:
                    alpha = TIFF.EXTRASAMPLE
                extrasamples = self.extrasamples
                if self.tags['ExtraSamples'].count == 1:
                    extrasamples = (extrasamples,)
                for i, exs in enumerate(extrasamples):
                    if exs in alpha:
                        if self.planarconfig == 1:
                            data = data[..., [0, 1, 2, 3 + i]]
                        else:
                            data = data[:, [0, 1, 2, 3 + i]]
                        break
            elif self.planarconfig == 1:
                data = data[..., :3]
            else:
                data = data[:, :3]
        elif photometric == PHOTOMETRIC.MINISBLACK:
            raise NotImplementedError()
        elif photometric == PHOTOMETRIC.MINISWHITE:
            raise NotImplementedError()
        elif photometric == PHOTOMETRIC.SEPARATED:
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        return data

    def aspage(self):
        return self

    @property
    def keyframe(self):
        return self

    @keyframe.setter
    def keyframe(self, index):
        return

    @lazyattr
    def offsets_bytecounts(self):
        """Return simplified offsets and bytecounts."""
        if self.is_contiguous:
            offset, byte_count = self.is_contiguous
            return ([offset], [byte_count])
        return clean_offsets_counts(self.dataoffsets, self.databytecounts)

    @lazyattr
    def is_contiguous(self):
        """Return offset and size of contiguous data, else None.

        Excludes prediction and fill_order.

        """
        if self.compression != 1 or self.bitspersample not in (8, 16, 32, 64):
            return
        if 'TileWidth' in self.tags:
            if self.imagewidth != self.tilewidth or self.imagelength % self.tilelength or self.tilewidth % 16 or self.tilelength % 16:
                return
            if 'ImageDepth' in self.tags and 'TileDepth' in self.tags and (self.imagelength != self.tilelength or self.imagedepth % self.tiledepth):
                return
        offsets = self.dataoffsets
        bytecounts = self.databytecounts
        if len(offsets) == 1:
            return (offsets[0], bytecounts[0])
        if self.is_stk or all((offsets[i] + bytecounts[i] == offsets[i + 1] or bytecounts[i + 1] == 0 for i in range(len(offsets) - 1))):
            return (offsets[0], sum(bytecounts))

    @lazyattr
    def is_final(self):
        """Return if page's image data are stored in final form.

        Excludes byte-swapping.

        """
        return self.is_contiguous and self.fillorder == 1 and (self.predictor == 1) and (not self.is_chroma_subsampled)

    @lazyattr
    def is_memmappable(self):
        """Return if page's image data in file can be memory-mapped."""
        return self.parent.filehandle.is_file and self.is_final and (self.is_contiguous[0] % self.dtype.itemsize == 0)

    def __str__(self, detail=0, width=79):
        """Return string containing information about page."""
        if self.keyframe != self:
            return TiffFrame.__str__(self, detail)
        attr = ''
        for name in ('memmappable', 'final', 'contiguous'):
            attr = getattr(self, 'is_' + name)
            if attr:
                attr = name.upper()
                break
        info = '  '.join((s for s in ('x'.join((str(i) for i in self.shape)), '%s%s' % (TIFF.SAMPLEFORMAT(self.sampleformat).name, self.bitspersample), '|'.join((i for i in (TIFF.PHOTOMETRIC(self.photometric).name, 'TILED' if self.is_tiled else '', self.compression.name if self.compression != 1 else '', self.planarconfig.name if self.planarconfig != 1 else '', self.predictor.name if self.predictor != 1 else '', self.fillorder.name if self.fillorder != 1 else '') if i)), attr, '|'.join((f.upper() for f in self.flags))) if s))
        info = 'TiffPage %i @%i  %s' % (self.index, self.offset, info)
        if detail <= 0:
            return info
        info = [info]
        tags = self.tags
        tlines = []
        vlines = []
        for tag in sorted(tags.values(), key=lambda x: x.code):
            value = tag.__str__(width=width + 1)
            tlines.append(value[:width].strip())
            if detail > 1 and len(value) > width:
                name = tag.name.upper()
                if detail <= 2 and ('COUNTS' in name or 'OFFSETS' in name):
                    value = pformat(tag.value, width=width, height=detail * 4)
                else:
                    value = pformat(tag.value, width=width, height=detail * 12)
                vlines.append('%s\n%s' % (tag.name, value))
        info.append('\n'.join(tlines))
        if detail > 1:
            info.append('\n\n'.join(vlines))
        if detail > 3:
            try:
                info.append('DATA\n%s' % pformat(self.asarray(), width=width, height=detail * 8))
            except Exception:
                pass
        return '\n\n'.join(info)

    @lazyattr
    def flags(self):
        """Return set of flags."""
        return set((name.lower() for name in sorted(TIFF.FILE_FLAGS) if getattr(self, 'is_' + name)))

    @property
    def ndim(self):
        """Return number of array dimensions."""
        return len(self.shape)

    @property
    def size(self):
        """Return number of elements in array."""
        return product(self.shape)

    @lazyattr
    def andor_tags(self):
        """Return consolidated metadata from Andor tags as dict.

        Remove Andor tags from self.tags.

        """
        if not self.is_andor:
            return
        tags = self.tags
        result = {'Id': tags['AndorId'].value}
        for tag in list(self.tags.values()):
            code = tag.code
            if not 4864 < code < 5031:
                continue
            value = tag.value
            name = tag.name[5:] if len(tag.name) > 5 else tag.name
            result[name] = value
            del tags[tag.name]
        return result

    @lazyattr
    def epics_tags(self):
        """Return consolidated metadata from EPICS areaDetector tags as dict.

        Remove areaDetector tags from self.tags.

        """
        if not self.is_epics:
            return
        result = {}
        tags = self.tags
        for tag in list(self.tags.values()):
            code = tag.code
            if not 65000 <= code < 65500:
                continue
            value = tag.value
            if code == 65000:
                result['timeStamp'] = datetime.datetime.fromtimestamp(float(value))
            elif code == 65001:
                result['uniqueID'] = int(value)
            elif code == 65002:
                result['epicsTSSec'] = int(value)
            elif code == 65003:
                result['epicsTSNsec'] = int(value)
            else:
                key, value = value.split(':', 1)
                result[key] = astype(value)
            del tags[tag.name]
        return result

    @lazyattr
    def geotiff_tags(self):
        """Return consolidated metadata from GeoTIFF tags as dict."""
        if not self.is_geotiff:
            return
        tags = self.tags
        gkd = tags['GeoKeyDirectoryTag'].value
        if gkd[0] != 1:
            warnings.warn('invalid GeoKeyDirectoryTag')
            return {}
        result = {'KeyDirectoryVersion': gkd[0], 'KeyRevision': gkd[1], 'KeyRevisionMinor': gkd[2]}
        geokeys = TIFF.GEO_KEYS
        geocodes = TIFF.GEO_CODES
        for index in range(gkd[3]):
            keyid, tagid, count, offset = gkd[4 + index * 4:index * 4 + 8]
            keyid = geokeys.get(keyid, keyid)
            if tagid == 0:
                value = offset
            else:
                tagname = TIFF.TAGS[tagid]
                value = tags[tagname].value[offset:offset + count]
                if tagid == 34737 and count > 1 and (value[-1] == '|'):
                    value = value[:-1]
                value = value if count > 1 else value[0]
            if keyid in geocodes:
                try:
                    value = geocodes[keyid](value)
                except Exception:
                    pass
            result[keyid] = value
        if 'IntergraphMatrixTag' in tags:
            value = tags['IntergraphMatrixTag'].value
            value = numpy.array(value)
            if len(value) == 16:
                value = value.reshape((4, 4)).tolist()
            result['IntergraphMatrix'] = value
        if 'ModelPixelScaleTag' in tags:
            value = numpy.array(tags['ModelPixelScaleTag'].value).tolist()
            result['ModelPixelScale'] = value
        if 'ModelTiepointTag' in tags:
            value = tags['ModelTiepointTag'].value
            value = numpy.array(value).reshape((-1, 6)).squeeze().tolist()
            result['ModelTiepoint'] = value
        if 'ModelTransformationTag' in tags:
            value = tags['ModelTransformationTag'].value
            value = numpy.array(value).reshape((4, 4)).tolist()
            result['ModelTransformation'] = value
        elif False:
            sx, sy, sz = tags['ModelPixelScaleTag'].value
            tiepoints = tags['ModelTiepointTag'].value
            transforms = []
            for tp in range(0, len(tiepoints), 6):
                i, j, k, x, y, z = tiepoints[tp:tp + 6]
                transforms.append([[sx, 0.0, 0.0, x - i * sx], [0.0, -sy, 0.0, y + j * sy], [0.0, 0.0, sz, z - k * sz], [0.0, 0.0, 0.0, 1.0]])
            if len(tiepoints) == 6:
                transforms = transforms[0]
            result['ModelTransformation'] = transforms
        if 'RPCCoefficientTag' in tags:
            rpcc = tags['RPCCoefficientTag'].value
            result['RPCCoefficient'] = {'ERR_BIAS': rpcc[0], 'ERR_RAND': rpcc[1], 'LINE_OFF': rpcc[2], 'SAMP_OFF': rpcc[3], 'LAT_OFF': rpcc[4], 'LONG_OFF': rpcc[5], 'HEIGHT_OFF': rpcc[6], 'LINE_SCALE': rpcc[7], 'SAMP_SCALE': rpcc[8], 'LAT_SCALE': rpcc[9], 'LONG_SCALE': rpcc[10], 'HEIGHT_SCALE': rpcc[11], 'LINE_NUM_COEFF': rpcc[12:33], 'LINE_DEN_COEFF ': rpcc[33:53], 'SAMP_NUM_COEFF': rpcc[53:73], 'SAMP_DEN_COEFF': rpcc[73:]}
        return result

    @property
    def is_tiled(self):
        """Page contains tiled image."""
        return 'TileWidth' in self.tags

    @property
    def is_reduced(self):
        """Page is reduced image of another image."""
        return 'NewSubfileType' in self.tags and self.tags['NewSubfileType'].value & 1

    @property
    def is_chroma_subsampled(self):
        """Page contains chroma subsampled image."""
        return 'YCbCrSubSampling' in self.tags and self.tags['YCbCrSubSampling'].value != (1, 1)

    @lazyattr
    def is_imagej(self):
        """Return ImageJ description if exists, else None."""
        for description in (self.description, self.description1):
            if not description:
                return
            if description[:7] == 'ImageJ=':
                return description

    @lazyattr
    def is_shaped(self):
        """Return description containing array shape if exists, else None."""
        for description in (self.description, self.description1):
            if not description:
                return
            if description[:1] == '{' and '"shape":' in description:
                return description
            if description[:6] == 'shape=':
                return description

    @property
    def is_mdgel(self):
        """Page contains MDFileTag tag."""
        return 'MDFileTag' in self.tags

    @property
    def is_mediacy(self):
        """Page contains Media Cybernetics Id tag."""
        return 'MC_Id' in self.tags and self.tags['MC_Id'].value[:7] == b'MC TIFF'

    @property
    def is_stk(self):
        """Page contains UIC2Tag tag."""
        return 'UIC2tag' in self.tags

    @property
    def is_lsm(self):
        """Page contains CZ_LSMINFO tag."""
        return 'CZ_LSMINFO' in self.tags

    @property
    def is_fluoview(self):
        """Page contains FluoView MM_STAMP tag."""
        return 'MM_Stamp' in self.tags

    @property
    def is_nih(self):
        """Page contains NIH image header."""
        return 'NIHImageHeader' in self.tags

    @property
    def is_sgi(self):
        """Page contains SGI image and tile depth tags."""
        return 'ImageDepth' in self.tags and 'TileDepth' in self.tags

    @property
    def is_vista(self):
        """Software tag is 'ISS Vista'."""
        return self.software == 'ISS Vista'

    @property
    def is_metaseries(self):
        """Page contains MDS MetaSeries metadata in ImageDescription tag."""
        if self.index > 1 or self.software != 'MetaSeries':
            return False
        d = self.description
        return d.startswith('<MetaData>') and d.endswith('</MetaData>')

    @property
    def is_ome(self):
        """Page contains OME-XML in ImageDescription tag."""
        if self.index > 1 or not self.description:
            return False
        d = self.description
        return d[:14] == '<?xml version=' and d[-6:] == '</OME>'

    @property
    def is_scn(self):
        """Page contains Leica SCN XML in ImageDescription tag."""
        if self.index > 1 or not self.description:
            return False
        d = self.description
        return d[:14] == '<?xml version=' and d[-6:] == '</scn>'

    @property
    def is_micromanager(self):
        """Page contains Micro-Manager metadata."""
        return 'MicroManagerMetadata' in self.tags

    @property
    def is_andor(self):
        """Page contains Andor Technology tags."""
        return 'AndorId' in self.tags

    @property
    def is_pilatus(self):
        """Page contains Pilatus tags."""
        return self.software[:8] == 'TVX TIFF' and self.description[:2] == '# '

    @property
    def is_epics(self):
        """Page contains EPICS areaDetector tags."""
        return self.description == 'EPICS areaDetector' or self.software == 'EPICS areaDetector'

    @property
    def is_tvips(self):
        """Page contains TVIPS metadata."""
        return 'TVIPS' in self.tags

    @property
    def is_fei(self):
        """Page contains SFEG or HELIOS metadata."""
        return 'FEI_SFEG' in self.tags or 'FEI_HELIOS' in self.tags

    @property
    def is_sem(self):
        """Page contains Zeiss SEM metadata."""
        return 'CZ_SEM' in self.tags

    @property
    def is_svs(self):
        """Page contains Aperio metadata."""
        return self.description[:20] == 'Aperio Image Library'

    @property
    def is_scanimage(self):
        """Page contains ScanImage metadata."""
        return self.description[:12] == 'state.config' or self.software[:22] == 'SI.LINE_FORMAT_VERSION' or 'scanimage.SI.' in self.description[-256:]

    @property
    def is_qptiff(self):
        """Page contains PerkinElmer tissue images metadata."""
        return self.software[:15] == 'PerkinElmer-QPI'

    @property
    def is_geotiff(self):
        """Page contains GeoTIFF metadata."""
        return 'GeoKeyDirectoryTag' in self.tags