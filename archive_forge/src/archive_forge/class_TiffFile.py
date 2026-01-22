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
class TiffFile(object):
    """Read image and metadata from TIFF file.

    TiffFile instances must be closed using the 'close' method, which is
    automatically called when using the 'with' context manager.

    Attributes
    ----------
    pages : TiffPages
        Sequence of TIFF pages in file.
    series : list of TiffPageSeries
        Sequences of closely related TIFF pages. These are computed
        from OME, LSM, ImageJ, etc. metadata or based on similarity
        of page properties such as shape, dtype, and compression.
    byteorder : '>', '<'
        The endianness of data in the file.
        '>': big-endian (Motorola).
        '>': little-endian (Intel).
    is_flag : bool
        If True, file is of a certain format.
        Flags are: bigtiff, movie, shaped, ome, imagej, stk, lsm, fluoview,
        nih, vista, 'micromanager, metaseries, mdgel, mediacy, tvips, fei,
        sem, scn, svs, scanimage, andor, epics, pilatus, qptiff.

    All attributes are read-only.

    Examples
    --------
    >>> # read image array from TIFF file
    >>> imsave('temp.tif', numpy.random.rand(5, 301, 219))
    >>> with TiffFile('temp.tif') as tif:
    ...     data = tif.asarray()
    >>> data.shape
    (5, 301, 219)

    """

    def __init__(self, arg, name=None, offset=None, size=None, multifile=True, movie=None, **kwargs):
        """Initialize instance from file.

        Parameters
        ----------
        arg : str or open file
            Name of file or open file object.
            The file objects are closed in TiffFile.close().
        name : str
            Optional name of file in case 'arg' is a file handle.
        offset : int
            Optional start position of embedded file. By default, this is
            the current file position.
        size : int
            Optional size of embedded file. By default, this is the number
            of bytes from the 'offset' to the end of the file.
        multifile : bool
            If True (default), series may include pages from multiple files.
            Currently applies to OME-TIFF only.
        movie : bool
            If True, assume that later pages differ from first page only by
            data offsets and byte counts. Significantly increases speed and
            reduces memory usage when reading movies with thousands of pages.
            Enabling this for non-movie files will result in data corruption
            or crashes. Python 3 only.
        kwargs : bool
            'is_ome': If False, disable processing of OME-XML metadata.

        """
        if 'fastij' in kwargs:
            del kwargs['fastij']
            raise DeprecationWarning('the fastij option will be removed')
        for key, value in kwargs.items():
            if key[:3] == 'is_' and key[3:] in TIFF.FILE_FLAGS:
                if value is not None and (not value):
                    setattr(self, key, bool(value))
            else:
                raise TypeError('unexpected keyword argument: %s' % key)
        fh = FileHandle(arg, mode='rb', name=name, offset=offset, size=size)
        self._fh = fh
        self._multifile = bool(multifile)
        self._files = {fh.name: self}
        try:
            fh.seek(0)
            try:
                byteorder = {b'II': '<', b'MM': '>'}[fh.read(2)]
            except KeyError:
                raise ValueError('not a TIFF file')
            sys_byteorder = {'big': '>', 'little': '<'}[sys.byteorder]
            self.isnative = byteorder == sys_byteorder
            version = struct.unpack(byteorder + 'H', fh.read(2))[0]
            if version == 43:
                self.is_bigtiff = True
                offsetsize, zero = struct.unpack(byteorder + 'HH', fh.read(4))
                if zero or offsetsize != 8:
                    raise ValueError('invalid BigTIFF file')
                self.byteorder = byteorder
                self.offsetsize = 8
                self.offsetformat = byteorder + 'Q'
                self.tagnosize = 8
                self.tagnoformat = byteorder + 'Q'
                self.tagsize = 20
                self.tagformat1 = byteorder + 'HH'
                self.tagformat2 = byteorder + 'Q8s'
            elif version == 42:
                self.is_bigtiff = False
                self.byteorder = byteorder
                self.offsetsize = 4
                self.offsetformat = byteorder + 'I'
                self.tagnosize = 2
                self.tagnoformat = byteorder + 'H'
                self.tagsize = 12
                self.tagformat1 = byteorder + 'HH'
                self.tagformat2 = byteorder + 'I4s'
            else:
                raise ValueError('invalid TIFF file')
            self.pages = TiffPages(self)
            if self.is_lsm and (self.filehandle.size >= 2 ** 32 or self.pages[0].compression != 1 or self.pages[1].compression != 1):
                self._lsm_load_pages()
                self._lsm_fix_strip_offsets()
                self._lsm_fix_strip_bytecounts()
            elif movie:
                self.pages.useframes = True
        except Exception:
            fh.close()
            raise

    @property
    def filehandle(self):
        """Return file handle."""
        return self._fh

    @property
    def filename(self):
        """Return name of file handle."""
        return self._fh.name

    @lazyattr
    def fstat(self):
        """Return status of file handle as stat_result object."""
        try:
            return os.fstat(self._fh.fileno())
        except Exception:
            return None

    def close(self):
        """Close open file handle(s)."""
        for tif in self._files.values():
            tif.filehandle.close()
        self._files = {}

    def asarray(self, key=None, series=None, out=None, validate=True, maxworkers=1):
        """Return image data from multiple TIFF pages as numpy array.

        By default, the data from the first series is returned.

        Parameters
        ----------
        key : int, slice, or sequence of page indices
            Defines which pages to return as array.
        series : int or TiffPageSeries
            Defines which series of pages to return as array.
        out : numpy.ndarray, str, or file-like object; optional
            Buffer where image data will be saved.
            If None (default), a new array will be created.
            If numpy.ndarray, a writable array of compatible dtype and shape.
            If 'memmap', directly memory-map the image data in the TIFF file
            if possible; else create a memory-mapped array in a temporary file.
            If str or open file, the file name or file object used to
            create a memory-map to an array stored in a binary file on disk.
        validate : bool
            If True (default), validate various tags.
            Passed to TiffPage.asarray().
        maxworkers : int
            Maximum number of threads to concurrently get data from pages.
            Default is 1. If None, up to half the CPU cores are used.
            Reading data from file is limited to a single thread.
            Using multiple threads can significantly speed up this function
            if the bottleneck is decoding compressed data, e.g. in case of
            large LZW compressed LSM files.
            If the bottleneck is I/O or pure Python code, using multiple
            threads might be detrimental.

        """
        if not self.pages:
            return numpy.array([])
        if key is None and series is None:
            series = 0
        if series is not None:
            try:
                series = self.series[series]
            except (KeyError, TypeError):
                pass
            pages = series._pages
        else:
            pages = self.pages
        if key is None:
            pass
        elif isinstance(key, inttypes):
            pages = [pages[key]]
        elif isinstance(key, slice):
            pages = pages[key]
        elif isinstance(key, collections.Iterable):
            pages = [pages[k] for k in key]
        else:
            raise TypeError('key must be an int, slice, or sequence')
        if not pages:
            raise ValueError('no pages selected')
        if self.is_nih:
            result = stack_pages(pages, out=out, maxworkers=maxworkers, squeeze=False)
        elif key is None and series and series.offset:
            typecode = self.byteorder + series.dtype.char
            if out == 'memmap' and pages[0].is_memmappable:
                result = self.filehandle.memmap_array(typecode, series.shape, series.offset)
            else:
                if out is not None:
                    out = create_output(out, series.shape, series.dtype)
                self.filehandle.seek(series.offset)
                result = self.filehandle.read_array(typecode, product(series.shape), out=out, native=True)
        elif len(pages) == 1:
            result = pages[0].asarray(out=out, validate=validate)
        else:
            result = stack_pages(pages, out=out, maxworkers=maxworkers)
        if result is None:
            return
        if key is None:
            try:
                result.shape = series.shape
            except ValueError:
                try:
                    warnings.warn('failed to reshape %s to %s' % (result.shape, series.shape))
                    result.shape = (-1,) + series.shape
                except ValueError:
                    result.shape = (-1,) + pages[0].shape
        elif len(pages) == 1:
            result.shape = pages[0].shape
        else:
            result.shape = (-1,) + pages[0].shape
        return result

    @lazyattr
    def series(self):
        """Return related pages as TiffPageSeries.

        Side effect: after calling this function, TiffFile.pages might contain
        TiffPage and TiffFrame instances.

        """
        if not self.pages:
            return []
        useframes = self.pages.useframes
        keyframe = self.pages.keyframe
        series = []
        for name in 'ome imagej lsm fluoview nih mdgel shaped'.split():
            if getattr(self, 'is_' + name, False):
                series = getattr(self, '_%s_series' % name)()
                break
        self.pages.useframes = useframes
        self.pages.keyframe = keyframe
        if not series:
            series = self._generic_series()
        series = [s for s in series if sum(s.shape) > 0]
        for i, s in enumerate(series):
            s.index = i
        return series

    def _generic_series(self):
        """Return image series in file."""
        if self.pages.useframes:
            page = self.pages[0]
            shape = page.shape
            axes = page.axes
            if len(self.pages) > 1:
                shape = (len(self.pages),) + shape
                axes = 'I' + axes
            return [TiffPageSeries(self.pages[:], shape, page.dtype, axes, stype='movie')]
        self.pages.clear(False)
        self.pages.load()
        result = []
        keys = []
        series = {}
        compressions = TIFF.DECOMPESSORS
        for page in self.pages:
            if not page.shape:
                continue
            key = page.shape + (page.axes, page.compression in compressions)
            if key in series:
                series[key].append(page)
            else:
                keys.append(key)
                series[key] = [page]
        for key in keys:
            pages = series[key]
            page = pages[0]
            shape = page.shape
            axes = page.axes
            if len(pages) > 1:
                shape = (len(pages),) + shape
                axes = 'I' + axes
            result.append(TiffPageSeries(pages, shape, page.dtype, axes, stype='Generic'))
        return result

    def _shaped_series(self):
        """Return image series in "shaped" file."""
        pages = self.pages
        pages.useframes = True
        lenpages = len(pages)

        def append_series(series, pages, axes, shape, reshape, name, truncated):
            page = pages[0]
            if not axes:
                shape = page.shape
                axes = page.axes
                if len(pages) > 1:
                    shape = (len(pages),) + shape
                    axes = 'Q' + axes
            size = product(shape)
            resize = product(reshape)
            if page.is_contiguous and resize > size and (resize % size == 0):
                if truncated is None:
                    truncated = True
                axes = 'Q' + axes
                shape = (resize // size,) + shape
            try:
                axes = reshape_axes(axes, shape, reshape)
                shape = reshape
            except ValueError as e:
                warnings.warn(str(e))
            series.append(TiffPageSeries(pages, shape, page.dtype, axes, name=name, stype='Shaped', truncated=truncated))
        keyframe = axes = shape = reshape = name = None
        series = []
        index = 0
        while True:
            if index >= lenpages:
                break
            pages.keyframe = index
            keyframe = pages[index]
            if not keyframe.is_shaped:
                warnings.warn('invalid shape metadata or corrupted file')
                return
            axes = None
            shape = None
            metadata = json_description_metadata(keyframe.is_shaped)
            name = metadata.get('name', '')
            reshape = metadata['shape']
            truncated = metadata.get('truncated', None)
            if 'axes' in metadata:
                axes = metadata['axes']
                if len(axes) == len(reshape):
                    shape = reshape
                else:
                    axes = ''
                    warnings.warn('axes do not match shape')
            spages = [keyframe]
            size = product(reshape)
            npages, mod = divmod(size, product(keyframe.shape))
            if mod:
                warnings.warn('series shape does not match page shape')
                return
            if 1 < npages <= lenpages - index:
                size *= keyframe._dtype.itemsize
                if truncated:
                    npages = 1
                elif keyframe.is_final and keyframe.offset + size < pages[index + 1].offset:
                    truncated = False
                else:
                    truncated = False
                    for j in range(index + 1, index + npages):
                        page = pages[j]
                        page.keyframe = keyframe
                        spages.append(page)
            append_series(series, spages, axes, shape, reshape, name, truncated)
            index += npages
        return series

    def _imagej_series(self):
        """Return image series in ImageJ file."""
        self.pages.useframes = True
        self.pages.keyframe = 0
        ij = self.imagej_metadata
        pages = self.pages
        page = pages[0]

        def is_hyperstack():
            if not page.is_final:
                return False
            images = ij.get('images', 0)
            if images <= 1:
                return False
            offset, count = page.is_contiguous
            if count != product(page.shape) * page.bitspersample // 8 or offset + count * images > self.filehandle.size:
                raise ValueError()
            if len(pages) > 1 and offset + count * images > pages[1].offset:
                return False
            return True
        try:
            hyperstack = is_hyperstack()
        except ValueError:
            warnings.warn('invalid ImageJ metadata or corrupted file')
            return
        if hyperstack:
            pages = [page]
        else:
            self.pages.load()
        shape = []
        axes = []
        if 'frames' in ij:
            shape.append(ij['frames'])
            axes.append('T')
        if 'slices' in ij:
            shape.append(ij['slices'])
            axes.append('Z')
        if 'channels' in ij and (not (page.photometric == 2 and (not ij.get('hyperstack', False)))):
            shape.append(ij['channels'])
            axes.append('C')
        remain = ij.get('images', len(pages)) // (product(shape) if shape else 1)
        if remain > 1:
            shape.append(remain)
            axes.append('I')
        if page.axes[0] == 'I':
            shape.extend(page.shape[1:])
            axes.extend(page.axes[1:])
        elif page.axes[:2] == 'SI':
            shape = page.shape[0:1] + tuple(shape) + page.shape[2:]
            axes = list(page.axes[0]) + axes + list(page.axes[2:])
        else:
            shape.extend(page.shape)
            axes.extend(page.axes)
        truncated = hyperstack and len(self.pages) == 1 and (page.is_contiguous[1] != product(shape) * page.bitspersample // 8)
        return [TiffPageSeries(pages, shape, page.dtype, axes, stype='ImageJ', truncated=truncated)]

    def _fluoview_series(self):
        """Return image series in FluoView file."""
        self.pages.useframes = True
        self.pages.keyframe = 0
        self.pages.load()
        mm = self.fluoview_metadata
        mmhd = list(reversed(mm['Dimensions']))
        axes = ''.join((TIFF.MM_DIMENSIONS.get(i[0].upper(), 'Q') for i in mmhd if i[1] > 1))
        shape = tuple((int(i[1]) for i in mmhd if i[1] > 1))
        return [TiffPageSeries(self.pages, shape, self.pages[0].dtype, axes, name=mm['ImageName'], stype='FluoView')]

    def _mdgel_series(self):
        """Return image series in MD Gel file."""
        self.pages.useframes = False
        self.pages.keyframe = 0
        self.pages.load()
        md = self.mdgel_metadata
        if md['FileTag'] in (2, 128):
            dtype = numpy.dtype('float32')
            scale = md['ScalePixel']
            scale = scale[0] / scale[1]
            if md['FileTag'] == 2:

                def transform(a):
                    return a.astype('float32') ** 2 * scale
            else:

                def transform(a):
                    return a.astype('float32') * scale
        else:
            transform = None
        page = self.pages[0]
        return [TiffPageSeries([page], page.shape, dtype, page.axes, transform=transform, stype='MDGel')]

    def _nih_series(self):
        """Return image series in NIH file."""
        self.pages.useframes = True
        self.pages.keyframe = 0
        self.pages.load()
        page0 = self.pages[0]
        if len(self.pages) == 1:
            shape = page0.shape
            axes = page0.axes
        else:
            shape = (len(self.pages),) + page0.shape
            axes = 'I' + page0.axes
        return [TiffPageSeries(self.pages, shape, page0.dtype, axes, stype='NIH')]

    def _ome_series(self):
        """Return image series in OME-TIFF file(s)."""
        from xml.etree import cElementTree as etree
        omexml = self.pages[0].description
        try:
            root = etree.fromstring(omexml)
        except etree.ParseError as e:
            warnings.warn('ome-xml: %s' % e)
            try:
                omexml = omexml.decode('utf-8', 'ignore').encode('utf-8')
                root = etree.fromstring(omexml)
            except Exception:
                return
        self.pages.useframes = True
        self.pages.keyframe = 0
        self.pages.load()
        uuid = root.attrib.get('UUID', None)
        self._files = {uuid: self}
        dirname = self._fh.dirname
        modulo = {}
        series = []
        for element in root:
            if element.tag.endswith('BinaryOnly'):
                warnings.warn('ome-xml: not an ome-tiff master file')
                break
            if element.tag.endswith('StructuredAnnotations'):
                for annot in element:
                    if not annot.attrib.get('Namespace', '').endswith('modulo'):
                        continue
                    for value in annot:
                        for modul in value:
                            for along in modul:
                                if not along.tag[:-1].endswith('Along'):
                                    continue
                                axis = along.tag[-1]
                                newaxis = along.attrib.get('Type', 'other')
                                newaxis = TIFF.AXES_LABELS[newaxis]
                                if 'Start' in along.attrib:
                                    step = float(along.attrib.get('Step', 1))
                                    start = float(along.attrib['Start'])
                                    stop = float(along.attrib['End']) + step
                                    labels = numpy.arange(start, stop, step)
                                else:
                                    labels = [label.text for label in along if label.tag.endswith('Label')]
                                modulo[axis] = (newaxis, labels)
            if not element.tag.endswith('Image'):
                continue
            attr = element.attrib
            name = attr.get('Name', None)
            for pixels in element:
                if not pixels.tag.endswith('Pixels'):
                    continue
                attr = pixels.attrib
                dtype = attr.get('PixelType', None)
                axes = ''.join(reversed(attr['DimensionOrder']))
                shape = list((int(attr['Size' + ax]) for ax in axes))
                size = product(shape[:-2])
                ifds = None
                spp = 1
                for data in pixels:
                    if data.tag.endswith('Channel'):
                        attr = data.attrib
                        if ifds is None:
                            spp = int(attr.get('SamplesPerPixel', spp))
                            ifds = [None] * (size // spp)
                        elif int(attr.get('SamplesPerPixel', 1)) != spp:
                            raise ValueError('cannot handle differing SamplesPerPixel')
                        continue
                    if ifds is None:
                        ifds = [None] * (size // spp)
                    if not data.tag.endswith('TiffData'):
                        continue
                    attr = data.attrib
                    ifd = int(attr.get('IFD', 0))
                    num = int(attr.get('NumPlanes', 1 if 'IFD' in attr else 0))
                    num = int(attr.get('PlaneCount', num))
                    idx = [int(attr.get('First' + ax, 0)) for ax in axes[:-2]]
                    try:
                        idx = numpy.ravel_multi_index(idx, shape[:-2])
                    except ValueError:
                        warnings.warn('ome-xml: invalid TiffData index')
                        continue
                    for uuid in data:
                        if not uuid.tag.endswith('UUID'):
                            continue
                        if uuid.text not in self._files:
                            if not self._multifile:
                                return []
                            fname = uuid.attrib['FileName']
                            try:
                                tif = TiffFile(os.path.join(dirname, fname))
                                tif.pages.useframes = True
                                tif.pages.keyframe = 0
                                tif.pages.load()
                            except (IOError, FileNotFoundError, ValueError):
                                warnings.warn("ome-xml: failed to read '%s'" % fname)
                                break
                            self._files[uuid.text] = tif
                            tif.close()
                        pages = self._files[uuid.text].pages
                        try:
                            for i in range(num if num else len(pages)):
                                ifds[idx + i] = pages[ifd + i]
                        except IndexError:
                            warnings.warn('ome-xml: index out of range')
                        break
                    else:
                        pages = self.pages
                        try:
                            for i in range(num if num else len(pages)):
                                ifds[idx + i] = pages[ifd + i]
                        except IndexError:
                            warnings.warn('ome-xml: index out of range')
                if all((i is None for i in ifds)):
                    continue
                keyframe = None
                for i in ifds:
                    if i and i == i.keyframe:
                        keyframe = i
                        break
                if not keyframe:
                    for i, keyframe in enumerate(ifds):
                        if keyframe:
                            keyframe.parent.pages.keyframe = keyframe.index
                            keyframe = keyframe.parent.pages[keyframe.index]
                            ifds[i] = keyframe
                            break
                for i in ifds:
                    if i is not None:
                        i.keyframe = keyframe
                dtype = keyframe.dtype
                series.append(TiffPageSeries(ifds, shape, dtype, axes, parent=self, name=name, stype='OME'))
        for serie in series:
            shape = list(serie.shape)
            for axis, (newaxis, labels) in modulo.items():
                i = serie.axes.index(axis)
                size = len(labels)
                if shape[i] == size:
                    serie.axes = serie.axes.replace(axis, newaxis, 1)
                else:
                    shape[i] //= size
                    shape.insert(i + 1, size)
                    serie.axes = serie.axes.replace(axis, axis + newaxis, 1)
            serie.shape = tuple(shape)
        for serie in series:
            serie.shape, serie.axes = squeeze_axes(serie.shape, serie.axes)
        return series

    def _lsm_series(self):
        """Return main image series in LSM file. Skip thumbnails."""
        lsmi = self.lsm_metadata
        axes = TIFF.CZ_LSMINFO_SCANTYPE[lsmi['ScanType']]
        if self.pages[0].photometric == 2:
            axes = axes.replace('C', '').replace('XY', 'XYC')
        if lsmi.get('DimensionP', 0) > 1:
            axes += 'P'
        if lsmi.get('DimensionM', 0) > 1:
            axes += 'M'
        axes = axes[::-1]
        shape = tuple((int(lsmi[TIFF.CZ_LSMINFO_DIMENSIONS[i]]) for i in axes))
        name = lsmi.get('Name', '')
        self.pages.keyframe = 0
        pages = self.pages[::2]
        dtype = pages[0].dtype
        series = [TiffPageSeries(pages, shape, dtype, axes, name=name, stype='LSM')]
        if self.pages[1].is_reduced:
            self.pages.keyframe = 1
            pages = self.pages[1::2]
            dtype = pages[0].dtype
            cp, i = (1, 0)
            while cp < len(pages) and i < len(shape) - 2:
                cp *= shape[i]
                i += 1
            shape = shape[:i] + pages[0].shape
            axes = axes[:i] + 'CYX'
            series.append(TiffPageSeries(pages, shape, dtype, axes, name=name, stype='LSMreduced'))
        return series

    def _lsm_load_pages(self):
        """Load all pages from LSM file."""
        self.pages.cache = True
        self.pages.useframes = True
        self.pages.keyframe = 1
        keyframe = self.pages[1]
        for page in self.pages[1::2]:
            page.keyframe = keyframe
        self.pages.keyframe = 0
        keyframe = self.pages[0]
        for page in self.pages[::2]:
            page.keyframe = keyframe

    def _lsm_fix_strip_offsets(self):
        """Unwrap strip offsets for LSM files greater than 4 GB.

        Each series and position require separate unwrapping (undocumented).

        """
        if self.filehandle.size < 2 ** 32:
            return
        pages = self.pages
        npages = len(pages)
        series = self.series[0]
        axes = series.axes
        positions = 1
        for i in (0, 1):
            if series.axes[i] in 'PM':
                positions *= series.shape[i]
        if positions > 1:
            ntimes = 0
            for i in (1, 2):
                if axes[i] == 'T':
                    ntimes = series.shape[i]
                    break
            if ntimes:
                div, mod = divmod(npages, 2 * positions * ntimes)
                assert mod == 0
                shape = (positions, ntimes, div, 2)
                indices = numpy.arange(product(shape)).reshape(shape)
                indices = numpy.moveaxis(indices, 1, 0)
        else:
            indices = numpy.arange(npages).reshape(-1, 2)
        if pages[0].dataoffsets[0] > pages[1].dataoffsets[0]:
            indices = indices[..., ::-1]
        wrap = 0
        previousoffset = 0
        for i in indices.flat:
            page = pages[i]
            dataoffsets = []
            for currentoffset in page.dataoffsets:
                if currentoffset < previousoffset:
                    wrap += 2 ** 32
                dataoffsets.append(currentoffset + wrap)
                previousoffset = currentoffset
            page.dataoffsets = tuple(dataoffsets)

    def _lsm_fix_strip_bytecounts(self):
        """Set databytecounts to size of compressed data.

        The StripByteCounts tag in LSM files contains the number of bytes
        for the uncompressed data.

        """
        pages = self.pages
        if pages[0].compression == 1:
            return
        pages = sorted(pages, key=lambda p: p.dataoffsets[0])
        npages = len(pages) - 1
        for i, page in enumerate(pages):
            if page.index % 2:
                continue
            offsets = page.dataoffsets
            bytecounts = page.databytecounts
            if i < npages:
                lastoffset = pages[i + 1].dataoffsets[0]
            else:
                lastoffset = min(offsets[-1] + 2 * bytecounts[-1], self._fh.size)
            offsets = offsets + (lastoffset,)
            page.databytecounts = tuple((offsets[j + 1] - offsets[j] for j in range(len(bytecounts))))

    def __getattr__(self, name):
        """Return 'is_flag' attributes from first page."""
        if name[3:] in TIFF.FILE_FLAGS:
            if not self.pages:
                return False
            value = bool(getattr(self.pages[0], name))
            setattr(self, name, value)
            return value
        raise AttributeError("'%s' object has no attribute '%s'" % (self.__class__.__name__, name))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __str__(self, detail=0, width=79):
        """Return string containing information about file.

        The detail parameter specifies the level of detail returned:

        0: file only.
        1: all series, first page of series and its tags.
        2: large tag values and file metadata.
        3: all pages.

        """
        info = ["TiffFile '%s'", format_size(self._fh.size), {'<': 'LittleEndian', '>': 'BigEndian'}[self.byteorder]]
        if self.is_bigtiff:
            info.append('BigTiff')
        info.append('|'.join((f.upper() for f in self.flags)))
        if len(self.pages) > 1:
            info.append('%i Pages' % len(self.pages))
        if len(self.series) > 1:
            info.append('%i Series' % len(self.series))
        if len(self._files) > 1:
            info.append('%i Files' % len(self._files))
        info = '  '.join(info)
        info = info.replace('    ', '  ').replace('   ', '  ')
        info = info % snipstr(self._fh.name, max(12, width + 2 - len(info)))
        if detail <= 0:
            return info
        info = [info]
        info.append('\n'.join((str(s) for s in self.series)))
        if detail >= 3:
            info.extend((TiffPage.__str__(p, detail=detail, width=width) for p in self.pages if p is not None))
        else:
            info.extend((TiffPage.__str__(s.pages[0], detail=detail, width=width) for s in self.series if s.pages[0] is not None))
        if detail >= 2:
            for name in sorted(self.flags):
                if hasattr(self, name + '_metadata'):
                    m = getattr(self, name + '_metadata')
                    if m:
                        info.append('%s_METADATA\n%s' % (name.upper(), pformat(m, width=width, height=detail * 12)))
        return '\n\n'.join(info).replace('\n\n\n', '\n\n')

    @lazyattr
    def flags(self):
        """Return set of file flags."""
        return set((name.lower() for name in sorted(TIFF.FILE_FLAGS) if getattr(self, 'is_' + name)))

    @lazyattr
    def is_mdgel(self):
        """File has MD Gel format."""
        try:
            return self.pages[0].is_mdgel or self.pages[1].is_mdgel
        except IndexError:
            return False

    @property
    def is_movie(self):
        """Return if file is a movie."""
        return self.pages.useframes

    @lazyattr
    def shaped_metadata(self):
        """Return Tifffile metadata from JSON descriptions as dicts."""
        if not self.is_shaped:
            return
        return tuple((json_description_metadata(s.pages[0].is_shaped) for s in self.series if s.stype.lower() == 'shaped'))

    @lazyattr
    def ome_metadata(self):
        """Return OME XML as dict."""
        if not self.is_ome:
            return
        return xml2dict(self.pages[0].description)['OME']

    @lazyattr
    def qptiff_metadata(self):
        """Return PerkinElmer-QPI-ImageDescription XML element as dict."""
        if not self.is_qptiff:
            return
        root = 'PerkinElmer-QPI-ImageDescription'
        xml = self.pages[0].description.replace(' ' + root + ' ', root)
        return xml2dict(xml)[root]

    @lazyattr
    def lsm_metadata(self):
        """Return LSM metadata from CZ_LSMINFO tag as dict."""
        if not self.is_lsm:
            return
        return self.pages[0].tags['CZ_LSMINFO'].value

    @lazyattr
    def stk_metadata(self):
        """Return STK metadata from UIC tags as dict."""
        if not self.is_stk:
            return
        page = self.pages[0]
        tags = page.tags
        result = {}
        result['NumberPlanes'] = tags['UIC2tag'].count
        if page.description:
            result['PlaneDescriptions'] = page.description.split('\x00')
        if 'UIC1tag' in tags:
            result.update(tags['UIC1tag'].value)
        if 'UIC3tag' in tags:
            result.update(tags['UIC3tag'].value)
        if 'UIC4tag' in tags:
            result.update(tags['UIC4tag'].value)
        uic2tag = tags['UIC2tag'].value
        result['ZDistance'] = uic2tag['ZDistance']
        result['TimeCreated'] = uic2tag['TimeCreated']
        result['TimeModified'] = uic2tag['TimeModified']
        try:
            result['DatetimeCreated'] = numpy.array([julian_datetime(*dt) for dt in zip(uic2tag['DateCreated'], uic2tag['TimeCreated'])], dtype='datetime64[ns]')
            result['DatetimeModified'] = numpy.array([julian_datetime(*dt) for dt in zip(uic2tag['DateModified'], uic2tag['TimeModified'])], dtype='datetime64[ns]')
        except ValueError as e:
            warnings.warn('stk_metadata: %s' % e)
        return result

    @lazyattr
    def imagej_metadata(self):
        """Return consolidated ImageJ metadata as dict."""
        if not self.is_imagej:
            return
        page = self.pages[0]
        result = imagej_description_metadata(page.is_imagej)
        if 'IJMetadata' in page.tags:
            try:
                result.update(page.tags['IJMetadata'].value)
            except Exception:
                pass
        return result

    @lazyattr
    def fluoview_metadata(self):
        """Return consolidated FluoView metadata as dict."""
        if not self.is_fluoview:
            return
        result = {}
        page = self.pages[0]
        result.update(page.tags['MM_Header'].value)
        result['Stamp'] = page.tags['MM_Stamp'].value
        return result

    @lazyattr
    def nih_metadata(self):
        """Return NIH Image metadata from NIHImageHeader tag as dict."""
        if not self.is_nih:
            return
        return self.pages[0].tags['NIHImageHeader'].value

    @lazyattr
    def fei_metadata(self):
        """Return FEI metadata from SFEG or HELIOS tags as dict."""
        if not self.is_fei:
            return
        tags = self.pages[0].tags
        if 'FEI_SFEG' in tags:
            return tags['FEI_SFEG'].value
        if 'FEI_HELIOS' in tags:
            return tags['FEI_HELIOS'].value

    @lazyattr
    def sem_metadata(self):
        """Return SEM metadata from CZ_SEM tag as dict."""
        if not self.is_sem:
            return
        return self.pages[0].tags['CZ_SEM'].value

    @lazyattr
    def mdgel_metadata(self):
        """Return consolidated metadata from MD GEL tags as dict."""
        for page in self.pages[:2]:
            if 'MDFileTag' in page.tags:
                tags = page.tags
                break
        else:
            return
        result = {}
        for code in range(33445, 33453):
            name = TIFF.TAGS[code]
            if name not in tags:
                continue
            result[name[2:]] = tags[name].value
        return result

    @lazyattr
    def andor_metadata(self):
        """Return Andor tags as dict."""
        return self.pages[0].andor_tags

    @lazyattr
    def epics_metadata(self):
        """Return EPICS areaDetector tags as dict."""
        return self.pages[0].epics_tags

    @lazyattr
    def tvips_metadata(self):
        """Return TVIPS tag as dict."""
        if not self.is_tvips:
            return
        return self.pages[0].tags['TVIPS'].value

    @lazyattr
    def metaseries_metadata(self):
        """Return MetaSeries metadata from image description as dict."""
        if not self.is_metaseries:
            return
        return metaseries_description_metadata(self.pages[0].description)

    @lazyattr
    def pilatus_metadata(self):
        """Return Pilatus metadata from image description as dict."""
        if not self.is_pilatus:
            return
        return pilatus_description_metadata(self.pages[0].description)

    @lazyattr
    def micromanager_metadata(self):
        """Return consolidated MicroManager metadata as dict."""
        if not self.is_micromanager:
            return
        result = read_micromanager_metadata(self._fh)
        result.update(self.pages[0].tags['MicroManagerMetadata'].value)
        return result

    @lazyattr
    def scanimage_metadata(self):
        """Return ScanImage non-varying frame and ROI metadata as dict."""
        if not self.is_scanimage:
            return
        result = {}
        try:
            framedata, roidata = read_scanimage_metadata(self._fh)
            result['FrameData'] = framedata
            result.update(roidata)
        except ValueError:
            pass
        try:
            result['Description'] = scanimage_description_metadata(self.pages[0].description)
        except Exception as e:
            warnings.warn('scanimage_description_metadata failed: %s' % e)
        return result

    @property
    def geotiff_metadata(self):
        """Return GeoTIFF metadata from first page as dict."""
        if not self.is_geotiff:
            return
        return self.pages[0].geotiff_tags