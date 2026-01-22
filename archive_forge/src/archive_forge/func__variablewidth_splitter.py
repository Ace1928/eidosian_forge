import numpy as np
import numpy.core.numeric as nx
from numpy.compat import asbytes, asunicode
def _variablewidth_splitter(self, line):
    if self.comments is not None:
        line = line.split(self.comments)[0]
    if not line:
        return []
    slices = self.delimiter
    return [line[s] for s in slices]