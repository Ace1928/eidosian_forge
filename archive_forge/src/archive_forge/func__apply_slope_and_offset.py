import sys
import os
import struct
import logging
import numpy as np
def _apply_slope_and_offset(self, data):
    """
        If RescaleSlope and RescaleIntercept are present in the data,
        apply them. The data type of the data is changed if necessary.
        """
    slope, offset = (1, 0)
    needFloats, needApplySlopeOffset = (False, False)
    if 'RescaleSlope' in self:
        needApplySlopeOffset = True
        slope = self.RescaleSlope
    if 'RescaleIntercept' in self:
        needApplySlopeOffset = True
        offset = self.RescaleIntercept
    if int(slope) != slope or int(offset) != offset:
        needFloats = True
    if not needFloats:
        slope, offset = (int(slope), int(offset))
    if needApplySlopeOffset:
        if data.dtype in [np.float32, np.float64]:
            pass
        elif needFloats:
            data = data.astype(np.float32)
        else:
            minReq, maxReq = (data.min(), data.max())
            minReq = min([minReq, minReq * slope + offset, maxReq * slope + offset])
            maxReq = max([maxReq, minReq * slope + offset, maxReq * slope + offset])
            dtype = None
            if minReq < 0:
                maxReq = max([-minReq, maxReq])
                if maxReq < 2 ** 7:
                    dtype = np.int8
                elif maxReq < 2 ** 15:
                    dtype = np.int16
                elif maxReq < 2 ** 31:
                    dtype = np.int32
                else:
                    dtype = np.float32
            elif maxReq < 2 ** 8:
                dtype = np.int8
            elif maxReq < 2 ** 16:
                dtype = np.int16
            elif maxReq < 2 ** 32:
                dtype = np.int32
            else:
                dtype = np.float32
            if dtype != data.dtype:
                data = data.astype(dtype)
        data *= slope
        data += offset
    return data