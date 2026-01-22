from __future__ import annotations
import math
def _getsum(self):
    """Get sum of all pixels in each layer"""
    v = []
    for i in range(0, len(self.h), 256):
        layer_sum = 0.0
        for j in range(256):
            layer_sum += j * self.h[i + j]
        v.append(layer_sum)
    return v