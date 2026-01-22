import numpy as np
import nibabel as nib
from nibabel.streamlines import FORMATS
from nibabel.streamlines.header import Field
def _gen_straight_streamline(start, end, steps=3):
    coords = []
    for s, e in zip(start, end):
        coords.append(np.linspace(s, e, steps))
    return np.array(coords).T