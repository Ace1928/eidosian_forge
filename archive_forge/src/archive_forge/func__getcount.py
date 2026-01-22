from __future__ import annotations
import math
def _getcount(self):
    """Get total number of pixels in each layer"""
    return [sum(self.h[i:i + 256]) for i in range(0, len(self.h), 256)]