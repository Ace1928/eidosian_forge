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
class TiffWriter(object):
    """Write numpy arrays to TIFF file.

    TiffWriter instances must be closed using the 'close' method, which is
    automatically called when using the 'with' context manager.

    TiffWriter's main purpose is saving nD numpy array's as TIFF,
    not to create any possible TIFF format. Specifically, JPEG compression,
    SubIFDs, ExifIFD, or GPSIFD tags are not supported.

    Examples
    --------
    >>> # successively append images to BigTIFF file
    >>> data = numpy.random.rand(2, 5, 3, 301, 219)
    >>> with TiffWriter('temp.tif', bigtiff=True) as tif:
    ...     for i in range(data.shape[0]):
    ...         tif.save(data[i], compress=6, photometric='minisblack')

    """

    def __init__(self, file, bigtiff=False, byteorder=None, append=False, imagej=False):
        """Open a TIFF file for writing.

        An empty TIFF file is created if the file does not exist, else the
        file is overwritten with an empty TIFF file unless 'append'
        is true. Use bigtiff=True when creating files larger than 4 GB.

        Parameters
        ----------
        file : str, binary stream, or FileHandle
            File name or writable binary stream, such as an open file
            or BytesIO.
        bigtiff : bool
            If True, the BigTIFF format is used.
        byteorder : {'<', '>', '=', '|'}
            The endianness of the data in the file.
            By default, this is the system's native byte order.
        append : bool
            If True and 'file' is an existing standard TIFF file, image data
            and tags are appended to the file.
            Appending data may corrupt specifically formatted TIFF files
            such as LSM, STK, ImageJ, NIH, or FluoView.
        imagej : bool
            If True, write an ImageJ hyperstack compatible file.
            This format can handle data types uint8, uint16, or float32 and
            data shapes up to 6 dimensions in TZCYXS order.
            RGB images (S=3 or S=4) must be uint8.
            ImageJ's default byte order is big-endian but this implementation
            uses the system's native byte order by default.
            ImageJ does not support BigTIFF format or LZMA compression.
            The ImageJ file format is undocumented.

        """
        if append:
            try:
                with FileHandle(file, mode='rb', size=0) as fh:
                    pos = fh.tell()
                    try:
                        with TiffFile(fh) as tif:
                            if append != 'force' and any((getattr(tif, 'is_' + a) for a in ('lsm', 'stk', 'imagej', 'nih', 'fluoview', 'micromanager'))):
                                raise ValueError('file contains metadata')
                            byteorder = tif.byteorder
                            bigtiff = tif.is_bigtiff
                            self._ifdoffset = tif.pages.next_page_offset
                    except Exception as e:
                        raise ValueError('cannot append to file: %s' % str(e))
                    finally:
                        fh.seek(pos)
            except (IOError, FileNotFoundError):
                append = False
        if byteorder in (None, '=', '|'):
            byteorder = '<' if sys.byteorder == 'little' else '>'
        elif byteorder not in ('<', '>'):
            raise ValueError('invalid byteorder %s' % byteorder)
        if imagej and bigtiff:
            warnings.warn('writing incompatible BigTIFF ImageJ')
        self._byteorder = byteorder
        self._imagej = bool(imagej)
        self._truncate = False
        self._metadata = None
        self._colormap = None
        self._descriptionoffset = 0
        self._descriptionlen = 0
        self._descriptionlenoffset = 0
        self._tags = None
        self._shape = None
        self._datashape = None
        self._datadtype = None
        self._dataoffset = None
        self._databytecounts = None
        self._tagoffsets = None
        if bigtiff:
            self._bigtiff = True
            self._offsetsize = 8
            self._tagsize = 20
            self._tagnoformat = 'Q'
            self._offsetformat = 'Q'
            self._valueformat = '8s'
        else:
            self._bigtiff = False
            self._offsetsize = 4
            self._tagsize = 12
            self._tagnoformat = 'H'
            self._offsetformat = 'I'
            self._valueformat = '4s'
        if append:
            self._fh = FileHandle(file, mode='r+b', size=0)
            self._fh.seek(0, 2)
        else:
            self._fh = FileHandle(file, mode='wb', size=0)
            self._fh.write({'<': b'II', '>': b'MM'}[byteorder])
            if bigtiff:
                self._fh.write(struct.pack(byteorder + 'HHH', 43, 8, 0))
            else:
                self._fh.write(struct.pack(byteorder + 'H', 42))
            self._ifdoffset = self._fh.tell()
            self._fh.write(struct.pack(byteorder + self._offsetformat, 0))

    def save(self, data=None, shape=None, dtype=None, returnoffset=False, photometric=None, planarconfig=None, tile=None, contiguous=True, align=16, truncate=False, compress=0, rowsperstrip=None, predictor=False, colormap=None, description=None, datetime=None, resolution=None, software='tifffile.py', metadata={}, ijmetadata=None, extratags=()):
        """Write numpy array and tags to TIFF file.

        The data shape's last dimensions are assumed to be image depth,
        height (length), width, and samples.
        If a colormap is provided, the data's dtype must be uint8 or uint16
        and the data values are indices into the last dimension of the
        colormap.
        If 'shape' and 'dtype' are specified, an empty array is saved.
        This option cannot be used with compression or multiple tiles.
        Image data are written uncompressed in one strip per plane by default.
        Dimensions larger than 2 to 4 (depending on photometric mode, planar
        configuration, and SGI mode) are flattened and saved as separate pages.
        The SampleFormat and BitsPerSample tags are derived from the data type.

        Parameters
        ----------
        data : numpy.ndarray or None
            Input image array.
        shape : tuple or None
            Shape of the empty array to save. Used only if 'data' is None.
        dtype : numpy.dtype or None
            Data-type of the empty array to save. Used only if 'data' is None.
        returnoffset : bool
            If True and the image data in the file is memory-mappable, return
            the offset and number of bytes of the image data in the file.
        photometric : {'MINISBLACK', 'MINISWHITE', 'RGB', 'PALETTE', 'CFA'}
            The color space of the image data.
            By default, this setting is inferred from the data shape and the
            value of colormap.
            For CFA images, DNG tags must be specified in 'extratags'.
        planarconfig : {'CONTIG', 'SEPARATE'}
            Specifies if samples are stored contiguous or in separate planes.
            By default, this setting is inferred from the data shape.
            If this parameter is set, extra samples are used to store grayscale
            images.
            'CONTIG': last dimension contains samples.
            'SEPARATE': third last dimension contains samples.
        tile : tuple of int
            The shape (depth, length, width) of image tiles to write.
            If None (default), image data are written in strips.
            The tile length and width must be a multiple of 16.
            If the tile depth is provided, the SGI ImageDepth and TileDepth
            tags are used to save volume data.
            Unless a single tile is used, tiles cannot be used to write
            contiguous files.
            Few software can read the SGI format, e.g. MeVisLab.
        contiguous : bool
            If True (default) and the data and parameters are compatible with
            previous ones, if any, the image data are stored contiguously after
            the previous one. Parameters 'photometric' and 'planarconfig'
            are ignored. Parameters 'description', datetime', and 'extratags'
            are written to the first page of a contiguous series only.
        align : int
            Byte boundary on which to align the image data in the file.
            Default 16. Use mmap.ALLOCATIONGRANULARITY for memory-mapped data.
            Following contiguous writes are not aligned.
        truncate : bool
            If True, only write the first page including shape metadata if
            possible (uncompressed, contiguous, not tiled).
            Other TIFF readers will only be able to read part of the data.
        compress : int or 'LZMA', 'ZSTD'
            Values from 0 to 9 controlling the level of zlib compression.
            If 0 (default), data are written uncompressed.
            Compression cannot be used to write contiguous files.
            If 'LZMA' or 'ZSTD', LZMA or ZSTD compression is used, which is
            not available on all platforms.
        rowsperstrip : int
            The number of rows per strip used for compression.
            Uncompressed data are written in one strip per plane.
        predictor : bool
            If True, apply horizontal differencing to integer type images
            before compression.
        colormap : numpy.ndarray
            RGB color values for the corresponding data value.
            Must be of shape (3, 2**(data.itemsize*8)) and dtype uint16.
        description : str
            The subject of the image. Must be 7-bit ASCII. Cannot be used with
            the ImageJ format. Saved with the first page only.
        datetime : datetime
            Date and time of image creation in '%Y:%m:%d %H:%M:%S' format.
            If None (default), the current date and time is used.
            Saved with the first page only.
        resolution : (float, float[, str]) or ((int, int), (int, int)[, str])
            X and Y resolutions in pixels per resolution unit as float or
            rational numbers. A third, optional parameter specifies the
            resolution unit, which must be None (default for ImageJ),
            'INCH' (default), or 'CENTIMETER'.
        software : str
            Name of the software used to create the file. Must be 7-bit ASCII.
            Saved with the first page only.
        metadata : dict
            Additional meta data to be saved along with shape information
            in JSON or ImageJ formats in an ImageDescription tag.
            If None, do not write a second ImageDescription tag.
            Strings must be 7-bit ASCII. Saved with the first page only.
        ijmetadata : dict
            Additional meta data to be saved in application specific
            IJMetadata and IJMetadataByteCounts tags. Refer to the
            imagej_metadata_tags function for valid keys and values.
            Saved with the first page only.
        extratags : sequence of tuples
            Additional tags as [(code, dtype, count, value, writeonce)].

            code : int
                The TIFF tag Id.
            dtype : str
                Data type of items in 'value' in Python struct format.
                One of B, s, H, I, 2I, b, h, i, 2i, f, d, Q, or q.
            count : int
                Number of data values. Not used for string or byte string
                values.
            value : sequence
                'Count' values compatible with 'dtype'.
                Byte strings must contain count values of dtype packed as
                binary data.
            writeonce : bool
                If True, the tag is written to the first page only.

        """
        fh = self._fh
        byteorder = self._byteorder
        if data is None:
            if compress:
                raise ValueError('cannot save compressed empty file')
            datashape = shape
            datadtype = numpy.dtype(dtype).newbyteorder(byteorder)
            datadtypechar = datadtype.char
        else:
            data = numpy.asarray(data, byteorder + data.dtype.char, 'C')
            if data.size == 0:
                raise ValueError('cannot save empty array')
            datashape = data.shape
            datadtype = data.dtype
            datadtypechar = data.dtype.char
        returnoffset = returnoffset and datadtype.isnative
        bilevel = datadtypechar == '?'
        if bilevel:
            index = -1 if datashape[-1] > 1 else -2
            datasize = product(datashape[:index])
            if datashape[index] % 8:
                datasize *= datashape[index] // 8 + 1
            else:
                datasize *= datashape[index] // 8
        else:
            datasize = product(datashape) * datadtype.itemsize
        self._truncate = bool(truncate)
        if self._datashape:
            if not contiguous or self._datashape[1:] != datashape or self._datadtype != datadtype or (compress and self._tags) or tile or (not numpy.array_equal(colormap, self._colormap)):
                self._write_remaining_pages()
                self._write_image_description()
                self._truncate = False
                self._descriptionoffset = 0
                self._descriptionlenoffset = 0
                self._datashape = None
                self._colormap = None
                if self._imagej:
                    raise ValueError('ImageJ does not support non-contiguous data')
            else:
                self._datashape = (self._datashape[0] + 1,) + datashape
                if not compress:
                    offset = fh.tell()
                    if data is None:
                        fh.write_empty(datasize)
                    else:
                        fh.write_array(data)
                    if returnoffset:
                        return (offset, datasize)
                    return
        input_shape = datashape
        tagnoformat = self._tagnoformat
        valueformat = self._valueformat
        offsetformat = self._offsetformat
        offsetsize = self._offsetsize
        tagsize = self._tagsize
        MINISBLACK = TIFF.PHOTOMETRIC.MINISBLACK
        RGB = TIFF.PHOTOMETRIC.RGB
        CFA = TIFF.PHOTOMETRIC.CFA
        PALETTE = TIFF.PHOTOMETRIC.PALETTE
        CONTIG = TIFF.PLANARCONFIG.CONTIG
        SEPARATE = TIFF.PLANARCONFIG.SEPARATE
        if photometric is not None:
            photometric = enumarg(TIFF.PHOTOMETRIC, photometric)
        if planarconfig:
            planarconfig = enumarg(TIFF.PLANARCONFIG, planarconfig)
        if not compress:
            compress = False
            compresstag = 1
            predictor = False
        else:
            if isinstance(compress, (tuple, list)):
                compress, compresslevel = compress
            elif isinstance(compress, int):
                compress, compresslevel = ('ADOBE_DEFLATE', int(compress))
                if not 0 <= compresslevel <= 9:
                    raise ValueError('invalid compression level %s' % compress)
            else:
                compresslevel = None
            compress = compress.upper()
            compresstag = enumarg(TIFF.COMPRESSION, compress)
        if self._imagej:
            if compress in ('LZMA', 'ZSTD'):
                raise ValueError('ImageJ cannot handle LZMA or ZSTD compression')
            if description:
                warnings.warn('not writing description to ImageJ file')
                description = None
            volume = False
            if datadtypechar not in 'BHhf':
                raise ValueError('ImageJ does not support data type %s' % datadtypechar)
            ijrgb = photometric == RGB if photometric else None
            if datadtypechar not in 'B':
                ijrgb = False
            ijshape = imagej_shape(datashape, ijrgb)
            if ijshape[-1] in (3, 4):
                photometric = RGB
                if datadtypechar not in 'B':
                    raise ValueError('ImageJ does not support data type %s for RGB' % datadtypechar)
            elif photometric is None:
                photometric = MINISBLACK
                planarconfig = None
            if planarconfig == SEPARATE:
                raise ValueError('ImageJ does not support planar images')
            else:
                planarconfig = CONTIG if ijrgb else None
        if compress:
            if compresslevel is None:
                compressor, compresslevel = TIFF.COMPESSORS[compresstag]
            else:
                compressor, _ = TIFF.COMPESSORS[compresstag]
                compresslevel = int(compresslevel)
            if predictor:
                if datadtype.kind not in 'iu':
                    raise ValueError('prediction not implemented for %s' % datadtype)

                def compress(data, level=compresslevel):
                    diff = numpy.diff(data, axis=-2)
                    data = numpy.insert(diff, 0, data[..., 0, :], axis=-2)
                    return compressor(data, level)
            else:

                def compress(data, level=compresslevel):
                    return compressor(data, level)
        if colormap is not None:
            if datadtypechar not in 'BH':
                raise ValueError('invalid data dtype for palette mode')
            colormap = numpy.asarray(colormap, dtype=byteorder + 'H')
            if colormap.shape != (3, 2 ** (datadtype.itemsize * 8)):
                raise ValueError('invalid color map shape')
            self._colormap = colormap
        if tile:
            tile = tuple((int(i) for i in tile[:3]))
            volume = len(tile) == 3
            if len(tile) < 2 or tile[-1] % 16 or tile[-2] % 16 or any((i < 1 for i in tile)):
                raise ValueError('invalid tile shape')
        else:
            tile = ()
            volume = False
        datashape = reshape_nd(datashape, 3 if photometric == RGB else 2)
        shape = datashape
        ndim = len(datashape)
        samplesperpixel = 1
        extrasamples = 0
        if volume and ndim < 3:
            volume = False
        if colormap is not None:
            photometric = PALETTE
            planarconfig = None
        if photometric is None:
            photometric = MINISBLACK
            if bilevel:
                photometric = TIFF.PHOTOMETRIC.MINISWHITE
            elif planarconfig == CONTIG:
                if ndim > 2 and shape[-1] in (3, 4):
                    photometric = RGB
            elif planarconfig == SEPARATE:
                if volume and ndim > 3 and (shape[-4] in (3, 4)):
                    photometric = RGB
                elif ndim > 2 and shape[-3] in (3, 4):
                    photometric = RGB
            elif ndim > 2 and shape[-1] in (3, 4):
                photometric = RGB
            elif self._imagej:
                photometric = MINISBLACK
            elif volume and ndim > 3 and (shape[-4] in (3, 4)):
                photometric = RGB
            elif ndim > 2 and shape[-3] in (3, 4):
                photometric = RGB
        if planarconfig and len(shape) <= (3 if volume else 2):
            planarconfig = None
            photometric = MINISBLACK
        if photometric == RGB:
            if len(shape) < 3:
                raise ValueError('not a RGB(A) image')
            if len(shape) < 4:
                volume = False
            if planarconfig is None:
                if shape[-1] in (3, 4):
                    planarconfig = CONTIG
                elif shape[-4 if volume else -3] in (3, 4):
                    planarconfig = SEPARATE
                elif shape[-1] > shape[-4 if volume else -3]:
                    planarconfig = SEPARATE
                else:
                    planarconfig = CONTIG
            if planarconfig == CONTIG:
                datashape = (-1, 1) + shape[-4 if volume else -3:]
                samplesperpixel = datashape[-1]
            else:
                datashape = (-1,) + shape[-4 if volume else -3:] + (1,)
                samplesperpixel = datashape[1]
            if samplesperpixel > 3:
                extrasamples = samplesperpixel - 3
        elif photometric == CFA:
            if len(shape) != 2:
                raise ValueError('invalid CFA image')
            volume = False
            planarconfig = None
            datashape = (-1, 1) + shape[-2:] + (1,)
            if 50706 not in (et[0] for et in extratags):
                raise ValueError('must specify DNG tags for CFA image')
        elif planarconfig and len(shape) > (3 if volume else 2):
            if planarconfig == CONTIG:
                datashape = (-1, 1) + shape[-4 if volume else -3:]
                samplesperpixel = datashape[-1]
            else:
                datashape = (-1,) + shape[-4 if volume else -3:] + (1,)
                samplesperpixel = datashape[1]
            extrasamples = samplesperpixel - 1
        else:
            planarconfig = None
            while len(shape) > 2 and shape[-1] == 1:
                shape = shape[:-1]
            if len(shape) < 3:
                volume = False
            datashape = (-1, 1) + shape[-3 if volume else -2:] + (1,)
        assert len(datashape) in (5, 6)
        if len(datashape) == 5:
            datashape = datashape[:2] + (1,) + datashape[2:]
        if datashape[0] == -1:
            s0 = product(input_shape) // product(datashape[1:])
            datashape = (s0,) + datashape[1:]
        shape = datashape
        if data is not None:
            data = data.reshape(shape)
        if tile and (not volume):
            tile = (1, tile[-2], tile[-1])
        if photometric == PALETTE:
            if samplesperpixel != 1 or extrasamples or shape[1] != 1 or (shape[-1] != 1):
                raise ValueError('invalid data shape for palette mode')
        if photometric == RGB and samplesperpixel == 2:
            raise ValueError('not a RGB image (samplesperpixel=2)')
        if bilevel:
            if compress:
                raise ValueError('cannot save compressed bilevel image')
            if tile:
                raise ValueError('cannot save tiled bilevel image')
            if photometric not in (0, 1):
                raise ValueError('cannot save bilevel image as %s' % str(photometric))
            datashape = list(datashape)
            if datashape[-2] % 8:
                datashape[-2] = datashape[-2] // 8 + 1
            else:
                datashape[-2] = datashape[-2] // 8
            datashape = tuple(datashape)
            assert datasize == product(datashape)
            if data is not None:
                data = numpy.packbits(data, axis=-2)
                assert datashape[-2] == data.shape[-2]
        bytestr = bytes if sys.version[0] == '2' else lambda x: bytes(x, 'ascii') if isinstance(x, str) else x
        tags = []
        strip_or_tile = 'Tile' if tile else 'Strip'
        tagbytecounts = TIFF.TAG_NAMES[strip_or_tile + 'ByteCounts']
        tag_offsets = TIFF.TAG_NAMES[strip_or_tile + 'Offsets']
        self._tagoffsets = tag_offsets

        def pack(fmt, *val):
            return struct.pack(byteorder + fmt, *val)

        def addtag(code, dtype, count, value, writeonce=False):
            code = int(TIFF.TAG_NAMES.get(code, code))
            try:
                tifftype = TIFF.DATA_DTYPES[dtype]
            except KeyError:
                raise ValueError('unknown dtype %s' % dtype)
            rawcount = count
            if dtype == 's':
                value = bytestr(value) + b'\x00'
                count = rawcount = len(value)
                rawcount = value.find(b'\x00\x00')
                if rawcount < 0:
                    rawcount = count
                else:
                    rawcount += 1
                value = (value,)
            elif isinstance(value, bytes):
                dtsize = struct.calcsize(dtype)
                if len(value) % dtsize:
                    raise ValueError('invalid packed binary data')
                count = len(value) // dtsize
            if len(dtype) > 1:
                count *= int(dtype[:-1])
                dtype = dtype[-1]
            ifdentry = [pack('HH', code, tifftype), pack(offsetformat, rawcount)]
            ifdvalue = None
            if struct.calcsize(dtype) * count <= offsetsize:
                if isinstance(value, bytes):
                    ifdentry.append(pack(valueformat, value))
                elif count == 1:
                    if isinstance(value, (tuple, list, numpy.ndarray)):
                        value = value[0]
                    ifdentry.append(pack(valueformat, pack(dtype, value)))
                else:
                    ifdentry.append(pack(valueformat, pack(str(count) + dtype, *value)))
            else:
                ifdentry.append(pack(offsetformat, 0))
                if isinstance(value, bytes):
                    ifdvalue = value
                elif isinstance(value, numpy.ndarray):
                    assert value.size == count
                    assert value.dtype.char == dtype
                    ifdvalue = value.tostring()
                elif isinstance(value, (tuple, list)):
                    ifdvalue = pack(str(count) + dtype, *value)
                else:
                    ifdvalue = pack(dtype, value)
            tags.append((code, b''.join(ifdentry), ifdvalue, writeonce))

        def rational(arg, max_denominator=1000000):
            """ "Return nominator and denominator from float or two integers."""
            from fractions import Fraction
            try:
                f = Fraction.from_float(arg)
            except TypeError:
                f = Fraction(arg[0], arg[1])
            f = f.limit_denominator(max_denominator)
            return (f.numerator, f.denominator)
        if description:
            addtag('ImageDescription', 's', 0, description, writeonce=True)
        self._metadata = {} if not metadata else metadata.copy()
        if self._imagej:
            description = imagej_description(input_shape, shape[-1] in (3, 4), self._colormap is not None, **self._metadata)
        elif metadata or metadata == {}:
            if self._truncate:
                self._metadata.update(truncated=True)
            description = json_description(input_shape, **self._metadata)
        else:
            description = None
        if description:
            description = str2bytes(description, 'ascii')
            description += b'\x00' * 64
            self._descriptionlen = len(description)
            addtag('ImageDescription', 's', 0, description, writeonce=True)
        if software:
            addtag('Software', 's', 0, software, writeonce=True)
        if datetime is None:
            datetime = self._now()
        addtag('DateTime', 's', 0, datetime.strftime('%Y:%m:%d %H:%M:%S'), writeonce=True)
        addtag('Compression', 'H', 1, compresstag)
        if predictor:
            addtag('Predictor', 'H', 1, 2)
        addtag('ImageWidth', 'I', 1, shape[-2])
        addtag('ImageLength', 'I', 1, shape[-3])
        if tile:
            addtag('TileWidth', 'I', 1, tile[-1])
            addtag('TileLength', 'I', 1, tile[-2])
            if tile[0] > 1:
                addtag('ImageDepth', 'I', 1, shape[-4])
                addtag('TileDepth', 'I', 1, tile[0])
        addtag('NewSubfileType', 'I', 1, 0)
        if not bilevel:
            sampleformat = {'u': 1, 'i': 2, 'f': 3, 'c': 6}[datadtype.kind]
            addtag('SampleFormat', 'H', samplesperpixel, (sampleformat,) * samplesperpixel)
        addtag('PhotometricInterpretation', 'H', 1, photometric.value)
        if colormap is not None:
            addtag('ColorMap', 'H', colormap.size, colormap)
        addtag('SamplesPerPixel', 'H', 1, samplesperpixel)
        if bilevel:
            pass
        elif planarconfig and samplesperpixel > 1:
            addtag('PlanarConfiguration', 'H', 1, planarconfig.value)
            addtag('BitsPerSample', 'H', samplesperpixel, (datadtype.itemsize * 8,) * samplesperpixel)
        else:
            addtag('BitsPerSample', 'H', 1, datadtype.itemsize * 8)
        if extrasamples:
            if photometric == RGB and extrasamples == 1:
                addtag('ExtraSamples', 'H', 1, 1)
            else:
                addtag('ExtraSamples', 'H', extrasamples, (0,) * extrasamples)
        if resolution is not None:
            addtag('XResolution', '2I', 1, rational(resolution[0]))
            addtag('YResolution', '2I', 1, rational(resolution[1]))
            if len(resolution) > 2:
                unit = resolution[2]
                unit = 1 if unit is None else enumarg(TIFF.RESUNIT, unit)
            elif self._imagej:
                unit = 1
            else:
                unit = 2
            addtag('ResolutionUnit', 'H', 1, unit)
        elif not self._imagej:
            addtag('XResolution', '2I', 1, (1, 1))
            addtag('YResolution', '2I', 1, (1, 1))
            addtag('ResolutionUnit', 'H', 1, 1)
        if ijmetadata:
            for t in imagej_metadata_tags(ijmetadata, byteorder):
                addtag(*t)
        contiguous = not compress
        if tile:
            tiles = ((shape[2] + tile[0] - 1) // tile[0], (shape[3] + tile[1] - 1) // tile[1], (shape[4] + tile[2] - 1) // tile[2])
            numtiles = product(tiles) * shape[1]
            stripbytecounts = [product(tile) * shape[-1] * datadtype.itemsize] * numtiles
            addtag(tagbytecounts, offsetformat, numtiles, stripbytecounts)
            addtag(tag_offsets, offsetformat, numtiles, [0] * numtiles)
            contiguous = contiguous and product(tiles) == 1
            if not contiguous:
                chunk = numpy.empty(tile + (shape[-1],), dtype=datadtype)
        elif contiguous:
            if bilevel:
                stripbytecounts = [product(datashape[2:])] * shape[1]
            else:
                stripbytecounts = [product(datashape[2:]) * datadtype.itemsize] * shape[1]
            addtag(tagbytecounts, offsetformat, shape[1], stripbytecounts)
            addtag(tag_offsets, offsetformat, shape[1], [0] * shape[1])
            addtag('RowsPerStrip', 'I', 1, shape[-3])
        else:
            rowsize = product(shape[-2:]) * datadtype.itemsize
            if rowsperstrip is None:
                rowsperstrip = 65536 // rowsize
            if rowsperstrip < 1:
                rowsperstrip = 1
            elif rowsperstrip > shape[-3]:
                rowsperstrip = shape[-3]
            addtag('RowsPerStrip', 'I', 1, rowsperstrip)
            numstrips = (shape[-3] + rowsperstrip - 1) // rowsperstrip
            numstrips *= shape[1]
            stripbytecounts = [0] * numstrips
            addtag(tagbytecounts, offsetformat, numstrips, [0] * numstrips)
            addtag(tag_offsets, offsetformat, numstrips, [0] * numstrips)
        if data is None and (not contiguous):
            raise ValueError('cannot write non-contiguous empty file')
        for t in extratags:
            addtag(*t)
        tags = sorted(tags, key=lambda x: x[0])
        if not (self._bigtiff or self._imagej) and fh.tell() + datasize > 2 ** 31 - 1:
            raise ValueError('data too large for standard TIFF file')
        for pageindex in range(1 if contiguous else shape[0]):
            pos = fh.tell()
            if pos % 2:
                fh.write(b'\x00')
                pos += 1
            fh.seek(self._ifdoffset)
            fh.write(pack(offsetformat, pos))
            fh.seek(pos)
            fh.write(pack(tagnoformat, len(tags)))
            tag_offset = fh.tell()
            fh.write(b''.join((t[1] for t in tags)))
            self._ifdoffset = fh.tell()
            fh.write(pack(offsetformat, 0))
            for tagindex, tag in enumerate(tags):
                if tag[2]:
                    pos = fh.tell()
                    if pos % 2:
                        fh.write(b'\x00')
                        pos += 1
                    fh.seek(tag_offset + tagindex * tagsize + offsetsize + 4)
                    fh.write(pack(offsetformat, pos))
                    fh.seek(pos)
                    if tag[0] == tag_offsets:
                        stripoffsetsoffset = pos
                    elif tag[0] == tagbytecounts:
                        strip_bytecounts_offset = pos
                    elif tag[0] == 270 and tag[2].endswith(b'\x00\x00\x00\x00'):
                        self._descriptionoffset = pos
                        self._descriptionlenoffset = tag_offset + tagindex * tagsize + 4
                    fh.write(tag[2])
            data_offset = fh.tell()
            skip = align - data_offset % align
            fh.seek(skip, 1)
            data_offset += skip
            if contiguous:
                if data is None:
                    fh.write_empty(datasize)
                else:
                    fh.write_array(data)
            elif tile:
                if data is None:
                    fh.write_empty(numtiles * stripbytecounts[0])
                else:
                    stripindex = 0
                    for plane in data[pageindex]:
                        for tz in range(tiles[0]):
                            for ty in range(tiles[1]):
                                for tx in range(tiles[2]):
                                    c0 = min(tile[0], shape[2] - tz * tile[0])
                                    c1 = min(tile[1], shape[3] - ty * tile[1])
                                    c2 = min(tile[2], shape[4] - tx * tile[2])
                                    chunk[c0:, c1:, c2:] = 0
                                    chunk[:c0, :c1, :c2] = plane[tz * tile[0]:tz * tile[0] + c0, ty * tile[1]:ty * tile[1] + c1, tx * tile[2]:tx * tile[2] + c2]
                                    if compress:
                                        t = compress(chunk)
                                        fh.write(t)
                                        stripbytecounts[stripindex] = len(t)
                                        stripindex += 1
                                    else:
                                        fh.write_array(chunk)
                                        fh.flush()
            elif compress:
                assert data.shape[2] == 1
                numstrips = (shape[-3] + rowsperstrip - 1) // rowsperstrip
                stripindex = 0
                for plane in data[pageindex]:
                    for i in range(numstrips):
                        strip = plane[0, i * rowsperstrip:(i + 1) * rowsperstrip]
                        strip = compress(strip)
                        fh.write(strip)
                        stripbytecounts[stripindex] = len(strip)
                        stripindex += 1
            pos = fh.tell()
            for tagindex, tag in enumerate(tags):
                if tag[0] == tag_offsets:
                    if tag[2]:
                        fh.seek(stripoffsetsoffset)
                        strip_offset = data_offset
                        for size in stripbytecounts:
                            fh.write(pack(offsetformat, strip_offset))
                            strip_offset += size
                    else:
                        fh.seek(tag_offset + tagindex * tagsize + offsetsize + 4)
                        fh.write(pack(offsetformat, data_offset))
                elif tag[0] == tagbytecounts:
                    if compress:
                        if tag[2]:
                            fh.seek(strip_bytecounts_offset)
                            for size in stripbytecounts:
                                fh.write(pack(offsetformat, size))
                        else:
                            fh.seek(tag_offset + tagindex * tagsize + offsetsize + 4)
                            fh.write(pack(offsetformat, stripbytecounts[0]))
                    break
            fh.seek(pos)
            fh.flush()
            if pageindex == 0:
                tags = [tag for tag in tags if not tag[-1]]
        self._shape = shape
        self._datashape = (1,) + input_shape
        self._datadtype = datadtype
        self._dataoffset = data_offset
        self._databytecounts = stripbytecounts
        if contiguous:
            self._tags = tags
            if returnoffset:
                return (data_offset, sum(stripbytecounts))

    def _write_remaining_pages(self):
        """Write outstanding IFDs and tags to file."""
        if not self._tags or self._truncate:
            return
        fh = self._fh
        fhpos = fh.tell()
        if fhpos % 2:
            fh.write(b'\x00')
            fhpos += 1
        byteorder = self._byteorder
        offsetformat = self._offsetformat
        offsetsize = self._offsetsize
        tagnoformat = self._tagnoformat
        tagsize = self._tagsize
        dataoffset = self._dataoffset
        pagedatasize = sum(self._databytecounts)
        pageno = self._shape[0] * self._datashape[0] - 1

        def pack(fmt, *val):
            return struct.pack(byteorder + fmt, *val)
        ifd = io.BytesIO()
        ifd.write(pack(tagnoformat, len(self._tags)))
        tagoffset = ifd.tell()
        ifd.write(b''.join((t[1] for t in self._tags)))
        ifdoffset = ifd.tell()
        ifd.write(pack(offsetformat, 0))
        for tagindex, tag in enumerate(self._tags):
            offset2value = tagoffset + tagindex * tagsize + offsetsize + 4
            if tag[2]:
                pos = ifd.tell()
                if pos % 2:
                    ifd.write(b'\x00')
                    pos += 1
                ifd.seek(offset2value)
                try:
                    ifd.write(pack(offsetformat, pos + fhpos))
                except Exception:
                    if self._imagej:
                        warnings.warn('truncating ImageJ file')
                        self._truncate = True
                        return
                    raise ValueError('data too large for non-BigTIFF file')
                ifd.seek(pos)
                ifd.write(tag[2])
                if tag[0] == self._tagoffsets:
                    stripoffset2offset = offset2value
                    stripoffset2value = pos
            elif tag[0] == self._tagoffsets:
                stripoffset2offset = None
                stripoffset2value = offset2value
        if ifd.tell() % 2:
            ifd.write(b'\x00')
        pos = fh.tell()
        if not self._bigtiff and pos + ifd.tell() * pageno > 2 ** 32 - 256:
            if self._imagej:
                warnings.warn('truncating ImageJ file')
                self._truncate = True
                return
            raise ValueError('data too large for non-BigTIFF file')
        for _ in range(pageno):
            pos = fh.tell()
            fh.seek(self._ifdoffset)
            fh.write(pack(offsetformat, pos))
            fh.seek(pos)
            self._ifdoffset = pos + ifdoffset
            dataoffset += pagedatasize
            if stripoffset2offset is None:
                ifd.seek(stripoffset2value)
                ifd.write(pack(offsetformat, dataoffset))
            else:
                ifd.seek(stripoffset2offset)
                ifd.write(pack(offsetformat, pos + stripoffset2value))
                ifd.seek(stripoffset2value)
                stripoffset = dataoffset
                for size in self._databytecounts:
                    ifd.write(pack(offsetformat, stripoffset))
                    stripoffset += size
            fh.write(ifd.getvalue())
        self._tags = None
        self._datadtype = None
        self._dataoffset = None
        self._databytecounts = None

    def _write_image_description(self):
        """Write meta data to ImageDescription tag."""
        if not self._datashape or self._datashape[0] == 1 or self._descriptionoffset <= 0:
            return
        colormapped = self._colormap is not None
        if self._imagej:
            isrgb = self._shape[-1] in (3, 4)
            description = imagej_description(self._datashape, isrgb, colormapped, **self._metadata)
        else:
            description = json_description(self._datashape, **self._metadata)
        description = description.encode('utf-8')
        description = description[:self._descriptionlen - 1]
        pos = self._fh.tell()
        self._fh.seek(self._descriptionoffset)
        self._fh.write(description)
        self._fh.seek(self._descriptionlenoffset)
        self._fh.write(struct.pack(self._byteorder + self._offsetformat, len(description) + 1))
        self._fh.seek(pos)
        self._descriptionoffset = 0
        self._descriptionlenoffset = 0
        self._descriptionlen = 0

    def _now(self):
        """Return current date and time."""
        return datetime.datetime.now()

    def close(self):
        """Write remaining pages and close file handle."""
        if not self._truncate:
            self._write_remaining_pages()
        self._write_image_description()
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()