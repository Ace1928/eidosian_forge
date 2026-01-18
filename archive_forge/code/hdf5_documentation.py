from .. import utils
from .._lazyload import h5py
from .._lazyload import tables
from decorator import decorator
Read values from a HDF5 dataset.

    Parameters
    ----------
    dataset : tables.CArray or h5py.Dataset

    Returns
    -------
    data : np.ndarray
        Data read from HDF5 dataset
    