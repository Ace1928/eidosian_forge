from __future__ import division
import decimal
import math
import re
import struct
import sys
import warnings
from collections import OrderedDict
import numpy as np
from . import Qt, debug, getConfigOption, reload
from .metaarray import MetaArray
from .Qt import QT_LIB, QtCore, QtGui
from .util.cupy_helper import getCupy
from .util.numba_helper import getNumbaFunctions
def affineSlice(data, shape, origin, vectors, axes, order=1, returnCoords=False, **kargs):
    """
    Take a slice of any orientation through an array. This is useful for extracting sections of multi-dimensional arrays
    such as MRI images for viewing as 1D or 2D data.
    
    The slicing axes are aribtrary; they do not need to be orthogonal to the original data or even to each other. It is
    possible to use this function to extract arbitrary linear, rectangular, or parallelepiped shapes from within larger
    datasets. The original data is interpolated onto a new array of coordinates using either interpolateArray if order<2
    or scipy.ndimage.map_coordinates otherwise.
    
    For a graphical interface to this function, see :func:`ROI.getArrayRegion <pyqtgraph.ROI.getArrayRegion>`
    
    ==============  ====================================================================================================
    **Arguments:**
    *data*          (ndarray) the original dataset
    *shape*         the shape of the slice to take (Note the return value may have more dimensions than len(shape))
    *origin*        the location in the original dataset that will become the origin of the sliced data.
    *vectors*       list of unit vectors which point in the direction of the slice axes. Each vector must have the same 
                    length as *axes*. If the vectors are not unit length, the result will be scaled relative to the 
                    original data. If the vectors are not orthogonal, the result will be sheared relative to the 
                    original data.
    *axes*          The axes in the original dataset which correspond to the slice *vectors*
    *order*         The order of spline interpolation. Default is 1 (linear). See scipy.ndimage.map_coordinates
                    for more information.
    *returnCoords*  If True, return a tuple (result, coords) where coords is the array of coordinates used to select
                    values from the original dataset.
    *All extra keyword arguments are passed to scipy.ndimage.map_coordinates.*
    --------------------------------------------------------------------------------------------------------------------
    ==============  ====================================================================================================
    
    Note the following must be true: 
        
        | len(shape) == len(vectors) 
        | len(origin) == len(axes) == len(vectors[i])
        
    Example: start with a 4D fMRI data set, take a diagonal-planar slice out of the last 3 axes
        
        * data = array with dims (time, x, y, z) = (100, 40, 40, 40)
        * The plane to pull out is perpendicular to the vector (x,y,z) = (1,1,1) 
        * The origin of the slice will be at (x,y,z) = (40, 0, 0)
        * We will slice a 20x20 plane from each timepoint, giving a final shape (100, 20, 20)
        
    The call for this example would look like::
        
        affineSlice(data, shape=(20,20), origin=(40,0,0), vectors=((-1, 1, 0), (-1, 0, 1)), axes=(1,2,3))
    
    """
    x = affineSliceCoords(shape, origin, vectors, axes)
    trAx = list(range(data.ndim))
    for ax in axes:
        trAx.remove(ax)
    tr1 = tuple(axes) + tuple(trAx)
    data = data.transpose(tr1)
    if order > 1:
        try:
            import scipy.ndimage
        except ImportError:
            raise ImportError('Interpolating with order > 1 requires the scipy.ndimage module, but it could not be imported.')
        extraShape = data.shape[len(axes):]
        output = np.empty(tuple(shape) + extraShape, dtype=data.dtype)
        for inds in np.ndindex(*extraShape):
            ind = (Ellipsis,) + inds
            output[ind] = scipy.ndimage.map_coordinates(data[ind], x, order=order, **kargs)
    else:
        tr = tuple(range(1, x.ndim)) + (0,)
        output = interpolateArray(data, x.transpose(tr), order=order)
    tr = list(range(output.ndim))
    trb = []
    for i in range(min(axes)):
        ind = tr1.index(i) + (len(shape) - len(axes))
        tr.remove(ind)
        trb.append(ind)
    tr2 = tuple(trb + tr)
    output = output.transpose(tr2)
    if returnCoords:
        return (output, x)
    else:
        return output