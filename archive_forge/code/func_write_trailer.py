import numpy as np
from ase.io.eps import EPS
def write_trailer(self, fd, renderer):
    import matplotlib.image
    buf = renderer.buffer_rgba()
    array = np.frombuffer(buf, dtype=np.uint8).reshape(int(self.h), int(self.w), 4)
    matplotlib.image.imsave(fd, array, format='png')