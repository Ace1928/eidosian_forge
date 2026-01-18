import numpy as np
def to_painting(image, saturation=1.4, black=0.006):
    """ transforms any photo into some kind of painting """
    edges = sobel(image.mean(axis=2))
    darkening = black * (255 * np.dstack(3 * [edges]))
    painting = saturation * image - darkening
    return np.maximum(0, np.minimum(255, painting)).astype('uint8')