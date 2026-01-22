import os
import sys
import shutil
import time
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import jsonio
from ase.io.ulm import open as ulmopen
from ase.parallel import paropen, world, barrier
from ase.calculators.singlepoint import (SinglePointCalculator,
class UlmBundleBackend:
    """Backend for BundleTrajectories stored as ASE Ulm files."""

    def __init__(self, master, singleprecision):
        self.writesmall = master
        self.writelarge = master
        self.singleprecision = singleprecision
        self.integral_dtypes = ['uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32', 'uint64', 'int64']
        self.int_dtype = dict(((k, getattr(np, k)) for k in self.integral_dtypes))
        self.int_minval = dict(((k, np.iinfo(self.int_dtype[k]).min) for k in self.integral_dtypes))
        self.int_maxval = dict(((k, np.iinfo(self.int_dtype[k]).max) for k in self.integral_dtypes))
        self.int_itemsize = dict(((k, np.dtype(self.int_dtype[k]).itemsize) for k in self.integral_dtypes))

    def write_small(self, framedir, smalldata):
        """Write small data to be written jointly."""
        if self.writesmall:
            with ulmopen(os.path.join(framedir, 'smalldata.ulm'), 'w') as fd:
                fd.write(**smalldata)

    def write(self, framedir, name, data):
        """Write data to separate file."""
        if self.writelarge:
            shape = data.shape
            dtype = str(data.dtype)
            stored_as = dtype
            all_identical = False
            if np.issubdtype(data.dtype, np.integer):
                minval = data.min()
                maxval = data.max()
                all_identical = bool(minval == maxval)
                if all_identical:
                    data = int(data.flat[0])
                else:
                    for typ in self.integral_dtypes:
                        if minval >= self.int_minval[typ] and maxval <= self.int_maxval[typ] and (data.itemsize > self.int_itemsize[typ]):
                            stored_as = typ
                            data = data.astype(self.int_dtype[typ])
            elif data.dtype == np.float32 or data.dtype == np.float64:
                all_identical = bool(data.min() == data.max())
                if all_identical:
                    data = float(data.flat[0])
                elif data.dtype == np.float64 and self.singleprecision:
                    stored_as = 'float32'
                    data = data.astype(np.float32)
            fn = os.path.join(framedir, name + '.ulm')
            with ulmopen(fn, 'w') as fd:
                fd.write(shape=shape, dtype=dtype, stored_as=stored_as, all_identical=all_identical, data=data)

    def read_small(self, framedir):
        """Read small data."""
        with ulmopen(os.path.join(framedir, 'smalldata.ulm'), 'r') as fd:
            return fd.asdict()

    def read(self, framedir, name):
        """Read data from separate file."""
        fn = os.path.join(framedir, name + '.ulm')
        with ulmopen(fn, 'r') as fd:
            if fd.all_identical:
                data = np.zeros(fd.shape, dtype=getattr(np, fd.dtype)) + fd.data
            elif fd.dtype == fd.stored_as:
                data = fd.data
            else:
                data = fd.data.astype(getattr(np, fd.dtype))
        return data

    def read_info(self, framedir, name, split=None):
        """Read information about file contents without reading the data.

        Information is a dictionary containing as aminimum the shape and
        type.
        """
        fn = os.path.join(framedir, name + '.ulm')
        if split is None or os.path.exists(fn):
            with ulmopen(fn, 'r') as fd:
                info = dict()
                info['shape'] = fd.shape
                info['type'] = fd.dtype
                info['stored_as'] = fd.stored_as
                info['identical'] = fd.all_identical
            return info
        else:
            info = dict()
            for i in range(split):
                fn = os.path.join(framedir, name + '_' + str(i) + '.ulm')
                with ulmopen(fn, 'r') as fd:
                    if i == 0:
                        info['shape'] = list(fd.shape)
                        info['type'] = fd.dtype
                        info['stored_as'] = fd.stored_as
                        info['identical'] = fd.all_identical
                    else:
                        info['shape'][0] += fd.shape[0]
                        assert info['type'] == fd.dtype
                        info['identical'] = info['identical'] and fd.all_identical
            info['shape'] = tuple(info['shape'])
            return info

    def set_fragments(self, nfrag):
        self.nfrag = nfrag

    def read_split(self, framedir, name):
        """Read data from multiple files.

        Falls back to reading from single file if that is how data is stored.

        Returns the data and an object indicating if the data was really
        read from split files.  The latter object is False if not
        read from split files, but is an array of the segment length if
        split files were used.
        """
        data = []
        if os.path.exists(os.path.join(framedir, name + '.ulm')):
            return (self.read(framedir, name), False)
        for i in range(self.nfrag):
            suf = '_%d' % (i,)
            data.append(self.read(framedir, name + suf))
        seglengths = [len(d) for d in data]
        return (np.concatenate(data), seglengths)

    def close(self, log=None):
        """Close anything that needs to be closed by the backend.

        The default backend does nothing here.
        """
        pass