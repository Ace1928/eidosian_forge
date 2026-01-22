import copy
import math
import numpy
def AddPoint(self, pt):
    self.data.append(pt[1:])
    self.ptNames.append(pt[0])
    self.nPts += 1