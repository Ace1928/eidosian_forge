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
class TiffSequence(object):
    """Sequence of TIFF files.

    The image data in all files must match shape, dtype, etc.

    Attributes
    ----------
    files : list
        List of file names.
    shape : tuple
        Shape of image sequence. Excludes shape of image array.
    axes : str
        Labels of axes in shape.

    Examples
    --------
    >>> # read image stack from sequence of TIFF files
    >>> imsave('temp_C001T001.tif', numpy.random.rand(64, 64))
    >>> imsave('temp_C001T002.tif', numpy.random.rand(64, 64))
    >>> tifs = TiffSequence('temp_C001*.tif')
    >>> tifs.shape
    (1, 2)
    >>> tifs.axes
    'CT'
    >>> data = tifs.asarray()
    >>> data.shape
    (1, 2, 64, 64)

    """
    _patterns = {'axes': '\n            # matches Olympus OIF and Leica TIFF series\n            _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\\d{1,4}))\n            _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\\d{1,4}))?\n            _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\\d{1,4}))?\n            _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\\d{1,4}))?\n            _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\\d{1,4}))?\n            _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\\d{1,4}))?\n            _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\\d{1,4}))?\n            '}

    class ParseError(Exception):
        pass

    def __init__(self, files, imread=TiffFile, pattern='axes', *args, **kwargs):
        """Initialize instance from multiple files.

        Parameters
        ----------
        files : str, pathlib.Path, or sequence thereof
            Glob pattern or sequence of file names.
            Binary streams are not supported.
        imread : function or class
            Image read function or class with asarray function returning numpy
            array from single file.
        pattern : str
            Regular expression pattern that matches axes names and sequence
            indices in file names.
            By default, the pattern matches Olympus OIF and Leica TIFF series.

        """
        if isinstance(files, pathlib.Path):
            files = str(files)
        if isinstance(files, basestring):
            files = natural_sorted(glob.glob(files))
        files = list(files)
        if not files:
            raise ValueError('no files found')
        if isinstance(files[0], pathlib.Path):
            files = [str(pathlib.Path(f)) for f in files]
        elif not isinstance(files[0], basestring):
            raise ValueError('not a file name')
        self.files = files
        if hasattr(imread, 'asarray'):
            _imread = imread

            def imread(fname, *args, **kwargs):
                with _imread(fname) as im:
                    return im.asarray(*args, **kwargs)
        self.imread = imread
        self.pattern = self._patterns.get(pattern, pattern)
        try:
            self._parse()
            if not self.axes:
                self.axes = 'I'
        except self.ParseError:
            self.axes = 'I'
            self.shape = (len(files),)
            self._startindex = (0,)
            self._indices = tuple(((i,) for i in range(len(files))))

    def __str__(self):
        """Return string with information about image sequence."""
        return '\n'.join([self.files[0], ' size: %i' % len(self.files), ' axes: %s' % self.axes, ' shape: %s' % str(self.shape)])

    def __len__(self):
        return len(self.files)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        pass

    def asarray(self, out=None, *args, **kwargs):
        """Read image data from all files and return as numpy array.

        The args and kwargs parameters are passed to the imread function.

        Raise IndexError or ValueError if image shapes do not match.

        """
        im = self.imread(self.files[0], *args, **kwargs)
        shape = self.shape + im.shape
        result = create_output(out, shape, dtype=im.dtype)
        result = result.reshape(-1, *im.shape)
        for index, fname in zip(self._indices, self.files):
            index = [i - j for i, j in zip(index, self._startindex)]
            index = numpy.ravel_multi_index(index, self.shape)
            im = self.imread(fname, *args, **kwargs)
            result[index] = im
        result.shape = shape
        return result

    def _parse(self):
        """Get axes and shape from file names."""
        if not self.pattern:
            raise self.ParseError('invalid pattern')
        pattern = re.compile(self.pattern, re.IGNORECASE | re.VERBOSE)
        matches = pattern.findall(self.files[0])
        if not matches:
            raise self.ParseError('pattern does not match file names')
        matches = matches[-1]
        if len(matches) % 2:
            raise self.ParseError('pattern does not match axis name and index')
        axes = ''.join((m for m in matches[::2] if m))
        if not axes:
            raise self.ParseError('pattern does not match file names')
        indices = []
        for fname in self.files:
            matches = pattern.findall(fname)[-1]
            if axes != ''.join((m for m in matches[::2] if m)):
                raise ValueError('axes do not match within the image sequence')
            indices.append([int(m) for m in matches[1::2] if m])
        shape = tuple(numpy.max(indices, axis=0))
        startindex = tuple(numpy.min(indices, axis=0))
        shape = tuple((i - j + 1 for i, j in zip(shape, startindex)))
        if product(shape) != len(self.files):
            warnings.warn('files are missing. Missing data are zeroed')
        self.axes = axes.upper()
        self.shape = shape
        self._indices = indices
        self._startindex = startindex