import operator
from OpenGL.arrays import buffers
from OpenGL.raw.GL import _types 
from OpenGL.raw.GL.VERSION import GL_1_1
from OpenGL import constant, error
Get contiguous array from source

        source -- numpy Python array (or compatible object)
            for use as the data source.  If this is not a contiguous
            array of the given typeCode, a copy will be made,
            otherwise will just be returned unchanged.
        typeCode -- optional 1-character typeCode specifier for
            the numpy.array function.

        All gl*Pointer calls should use contiguous arrays, as non-
        contiguous arrays will be re-copied on every rendering pass.
        Although this doesn't raise an error, it does tend to slow
        down rendering.
        