import sys
import os
import random
import logging
import json
import warnings
from numbers import Number
import numpy as np
from .. import numpy as _mx_np  # pylint: disable=reimported
from ..base import numeric_types
from .. import ndarray as nd
from ..ndarray import _internal
from .. import io
from .. import recordio
from .. util import is_np_array
from ..ndarray.numpy import _internal as _npi
class ImageIter(io.DataIter):
    """Image data iterator with a large number of augmentation choices.
    This iterator supports reading from both .rec files and raw image files.

    To load input images from .rec files, use `path_imgrec` parameter and to load from raw image
    files, use `path_imglist` and `path_root` parameters.

    To use data partition (for distributed training) or shuffling, specify `path_imgidx` parameter.

    Parameters
    ----------
    batch_size : int
        Number of examples per batch.
    data_shape : tuple
        Data shape in (channels, height, width) format.
        For now, only RGB image with 3 channels is supported.
    label_width : int, optional
        Number of labels per example. The default label width is 1.
    path_imgrec : str
        Path to image record file (.rec).
        Created with tools/im2rec.py or bin/im2rec.
    path_imglist : str
        Path to image list (.lst).
        Created with tools/im2rec.py or with custom script.
        Format: Tab separated record of index, one or more labels and relative_path_from_root.
    imglist: list
        A list of images with the label(s).
        Each item is a list [imagelabel: float or list of float, imgpath].
    path_root : str
        Root folder of image files.
    path_imgidx : str
        Path to image index file. Needed for partition and shuffling when using .rec source.
    shuffle : bool
        Whether to shuffle all images at the start of each iteration or not.
        Can be slow for HDD.
    part_index : int
        Partition index.
    num_parts : int
        Total number of partitions.
    data_name : str
        Data name for provided symbols.
    label_name : str
        Label name for provided symbols.
    dtype : str
        Label data type. Default: float32. Other options: int32, int64, float64
    last_batch_handle : str, optional
        How to handle the last batch.
        This parameter can be 'pad'(default), 'discard' or 'roll_over'.
        If 'pad', the last batch will be padded with data starting from the begining
        If 'discard', the last batch will be discarded
        If 'roll_over', the remaining elements will be rolled over to the next iteration
    kwargs : ...
        More arguments for creating augmenter. See mx.image.CreateAugmenter.
    """

    def __init__(self, batch_size, data_shape, label_width=1, path_imgrec=None, path_imglist=None, path_root=None, path_imgidx=None, shuffle=False, part_index=0, num_parts=1, aug_list=None, imglist=None, data_name='data', label_name='softmax_label', dtype='float32', last_batch_handle='pad', **kwargs):
        super(ImageIter, self).__init__()
        assert path_imgrec or path_imglist or isinstance(imglist, list)
        assert dtype in ['int32', 'float32', 'int64', 'float64'], dtype + ' label not supported'
        num_threads = os.environ.get('MXNET_CPU_WORKER_NTHREADS', 1)
        logging.info('Using %s threads for decoding...', str(num_threads))
        logging.info('Set enviroment variable MXNET_CPU_WORKER_NTHREADS to a larger number to use more threads.')
        class_name = self.__class__.__name__
        if path_imgrec:
            logging.info('%s: loading recordio %s...', class_name, path_imgrec)
            if path_imgidx:
                self.imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
                self.imgidx = list(self.imgrec.keys)
            else:
                self.imgrec = recordio.MXRecordIO(path_imgrec, 'r')
                self.imgidx = None
        else:
            self.imgrec = None
        array_fn = _mx_np.array if is_np_array() else nd.array
        if path_imglist:
            logging.info('%s: loading image list %s...', class_name, path_imglist)
            with open(path_imglist) as fin:
                imglist = {}
                imgkeys = []
                for line in iter(fin.readline, ''):
                    line = line.strip().split('\t')
                    label = array_fn(line[1:-1], dtype=dtype)
                    key = int(line[0])
                    imglist[key] = (label, line[-1])
                    imgkeys.append(key)
                self.imglist = imglist
        elif isinstance(imglist, list):
            logging.info('%s: loading image list...', class_name)
            result = {}
            imgkeys = []
            index = 1
            for img in imglist:
                key = str(index)
                index += 1
                if len(img) > 2:
                    label = array_fn(img[:-1], dtype=dtype)
                elif isinstance(img[0], numeric_types):
                    label = array_fn([img[0]], dtype=dtype)
                else:
                    label = array_fn(img[0], dtype=dtype)
                result[key] = (label, img[-1])
                imgkeys.append(str(key))
            self.imglist = result
        else:
            self.imglist = None
        self.path_root = path_root
        self.check_data_shape(data_shape)
        self.provide_data = [(data_name, (batch_size,) + data_shape)]
        if label_width > 1:
            self.provide_label = [(label_name, (batch_size, label_width))]
        else:
            self.provide_label = [(label_name, (batch_size,))]
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.label_width = label_width
        self.shuffle = shuffle
        if self.imgrec is None:
            self.seq = imgkeys
        elif shuffle or num_parts > 1 or path_imgidx:
            assert self.imgidx is not None
            self.seq = self.imgidx
        else:
            self.seq = None
        if num_parts > 1:
            assert part_index < num_parts
            N = len(self.seq)
            C = N // num_parts
            self.seq = self.seq[part_index * C:(part_index + 1) * C]
        if aug_list is None:
            self.auglist = CreateAugmenter(data_shape, **kwargs)
        else:
            self.auglist = aug_list
        self.cur = 0
        self._allow_read = True
        self.last_batch_handle = last_batch_handle
        self.num_image = len(self.seq) if self.seq is not None else None
        self._cache_data = None
        self._cache_label = None
        self._cache_idx = None
        self.reset()

    def reset(self):
        """Resets the iterator to the beginning of the data."""
        if self.seq is not None and self.shuffle:
            random.shuffle(self.seq)
        if self.last_batch_handle != 'roll_over' or self._cache_data is None:
            if self.imgrec is not None:
                self.imgrec.reset()
            self.cur = 0
            if self._allow_read is False:
                self._allow_read = True

    def hard_reset(self):
        """Resets the iterator and ignore roll over data"""
        if self.seq is not None and self.shuffle:
            random.shuffle(self.seq)
        if self.imgrec is not None:
            self.imgrec.reset()
        self.cur = 0
        self._allow_read = True
        self._cache_data = None
        self._cache_label = None
        self._cache_idx = None

    def next_sample(self):
        """Helper function for reading in next sample."""
        if self._allow_read is False:
            raise StopIteration
        if self.seq is not None:
            if self.cur < self.num_image:
                idx = self.seq[self.cur]
            else:
                if self.last_batch_handle != 'discard':
                    self.cur = 0
                raise StopIteration
            self.cur += 1
            if self.imgrec is not None:
                s = self.imgrec.read_idx(idx)
                header, img = recordio.unpack(s)
                if self.imglist is None:
                    return (header.label, img)
                else:
                    return (self.imglist[idx][0], img)
            else:
                label, fname = self.imglist[idx]
                return (label, self.read_image(fname))
        else:
            s = self.imgrec.read()
            if s is None:
                if self.last_batch_handle != 'discard':
                    self.imgrec.reset()
                raise StopIteration
            header, img = recordio.unpack(s)
            return (header.label, img)

    def _batchify(self, batch_data, batch_label, start=0):
        """Helper function for batchifying data"""
        i = start
        batch_size = self.batch_size
        try:
            while i < batch_size:
                label, s = self.next_sample()
                data = self.imdecode(s)
                try:
                    self.check_valid_image(data)
                except RuntimeError as e:
                    logging.debug('Invalid image, skipping:  %s', str(e))
                    continue
                data = self.augmentation_transform(data)
                assert i < batch_size, 'Batch size must be multiples of augmenter output length'
                batch_data[i] = self.postprocess_data(data)
                batch_label[i] = label
                i += 1
        except StopIteration:
            if not i:
                raise StopIteration
        return i

    def next(self):
        """Returns the next batch of data."""
        batch_size = self.batch_size
        c, h, w = self.data_shape
        if self._cache_data is not None:
            assert self._cache_label is not None, "_cache_label didn't have values"
            assert self._cache_idx is not None, "_cache_idx didn't have values"
            batch_data = self._cache_data
            batch_label = self._cache_label
            i = self._cache_idx
        else:
            if is_np_array():
                zeros_fn = _mx_np.zeros
                empty_fn = _mx_np.empty
            else:
                zeros_fn = nd.zeros
                empty_fn = nd.empty
            batch_data = zeros_fn((batch_size, c, h, w))
            batch_label = empty_fn(self.provide_label[0][1])
            i = self._batchify(batch_data, batch_label)
        pad = batch_size - i
        if pad != 0:
            if self.last_batch_handle == 'discard':
                raise StopIteration
            if self.last_batch_handle == 'roll_over' and self._cache_data is None:
                self._cache_data = batch_data
                self._cache_label = batch_label
                self._cache_idx = i
                raise StopIteration
            _ = self._batchify(batch_data, batch_label, i)
            if self.last_batch_handle == 'pad':
                self._allow_read = False
            else:
                self._cache_data = None
                self._cache_label = None
                self._cache_idx = None
        return io.DataBatch([batch_data], [batch_label], pad=pad)

    def check_data_shape(self, data_shape):
        """Checks if the input data shape is valid"""
        if not len(data_shape) == 3:
            raise ValueError('data_shape should have length 3, with dimensions CxHxW')
        if not data_shape[0] == 3:
            raise ValueError('This iterator expects inputs to have 3 channels.')

    def check_valid_image(self, data):
        """Checks if the input data is valid"""
        if len(data[0].shape) == 0:
            raise RuntimeError('Data shape is wrong')

    def imdecode(self, s):
        """Decodes a string or byte string to an NDArray.
        See mx.img.imdecode for more details."""

        def locate():
            """Locate the image file/index if decode fails."""
            if self.seq is not None:
                idx = self.seq[self.cur % self.num_image - 1]
            else:
                idx = self.cur % self.num_image - 1
            if self.imglist is not None:
                _, fname = self.imglist[idx]
                msg = 'filename: {}'.format(fname)
            else:
                msg = 'index: {}'.format(idx)
            return 'Broken image ' + msg
        try:
            img = imdecode(s)
        except Exception as e:
            raise RuntimeError('{}, {}'.format(locate(), e))
        return img

    def read_image(self, fname):
        """Reads an input image `fname` and returns the decoded raw bytes.
        Examples
        --------
        >>> dataIter.read_image('Face.jpg') # returns decoded raw bytes.
        """
        with open(os.path.join(self.path_root, fname), 'rb') as fin:
            img = fin.read()
        return img

    def augmentation_transform(self, data):
        """Transforms input data with specified augmentation."""
        for aug in self.auglist:
            data = aug(data)
        return data

    def postprocess_data(self, datum):
        """Final postprocessing step before image is loaded into the batch."""
        if is_np_array():
            return datum.transpose(2, 0, 1)
        else:
            return nd.transpose(datum, axes=(2, 0, 1))