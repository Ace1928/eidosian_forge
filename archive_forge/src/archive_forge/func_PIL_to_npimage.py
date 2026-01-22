import numpy as np
def PIL_to_npimage(im):
    """ Transforms a PIL/Pillow image into a numpy RGB(A) image.
        Actually all this do is returning numpy.array(im)."""
    return np.array(im)