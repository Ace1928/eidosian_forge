import numpy
from rdkit.ML.Data import Quantize
def _getBinId(val, qBounds):
    bid = 0
    for bnd in qBounds:
        if val > bnd:
            bid += 1
    return bid