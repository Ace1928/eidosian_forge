import copy
import os
import pickle
import warnings
import numpy as np
def writeMeta(self, fileName):
    """Used to re-write meta info to the given file.
        This feature is only available for HDF5 files."""
    f = h5py.File(fileName, 'r+')
    if f.attrs['MetaArray'] != MetaArray.version:
        raise Exception('The file %s was created with a different version of MetaArray. Will not modify.' % fileName)
    del f['info']
    self.writeHDF5Meta(f, 'info', self._info)
    f.close()