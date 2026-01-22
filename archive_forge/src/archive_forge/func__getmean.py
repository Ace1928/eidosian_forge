from __future__ import annotations
import math
def _getmean(self):
    """Get average pixel level for each layer"""
    return [self.sum[i] / self.count[i] for i in self.bands]