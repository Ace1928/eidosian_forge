from __future__ import annotations
import math
def _getsum2(self):
    """Get squared sum of all pixels in each layer"""
    v = []
    for i in range(0, len(self.h), 256):
        sum2 = 0.0
        for j in range(256):
            sum2 += j ** 2 * float(self.h[i + j])
        v.append(sum2)
    return v