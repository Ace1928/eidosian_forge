import warnings
import numpy
def weaklyDominates(point, other):
    for i in range(len(point)):
        if point[i] > other[i]:
            return False
    return True