from . import encode
from . import number_types as N
def GetArrayAsNumpy(self, flags, off, length):
    """
        GetArrayAsNumpy returns the array with fixed width that starts at `Vector(offset)`
        with length `length` as a numpy array with the type specified by `flags`. The
        array is a `view` into Bytes so modifying the returned will modify Bytes in place.
        """
    numpy_dtype = N.to_numpy_type(flags)
    return encode.GetVectorAsNumpy(numpy_dtype, self.Bytes, length, off)