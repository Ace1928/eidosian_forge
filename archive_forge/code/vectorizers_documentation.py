from numba import cuda
from numpy import array as np_array
from numba.np.ufunc import deviceufunc
from numba.np.ufunc.deviceufunc import (UFuncMechanism, GeneralizedUFunc,

        *args: numpy arrays or DeviceArrayBase (created by cuda.to_device).
               Cannot mix the two types in one call.

        **kws:
            stream -- cuda stream; when defined, asynchronous mode is used.
            out    -- output array. Can be a numpy array or DeviceArrayBase
                      depending on the input arguments.  Type must match
                      the input arguments.
        