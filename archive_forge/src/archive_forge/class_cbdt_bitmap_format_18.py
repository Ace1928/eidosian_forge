from fontTools.misc.textTools import bytesjoin
from fontTools.misc import sstruct
from . import E_B_D_T_
from .BitmapGlyphMetrics import (
from .E_B_D_T_ import (
import struct
class cbdt_bitmap_format_18(BitmapPlusBigMetricsMixin, ColorBitmapGlyph):

    def decompile(self):
        self.metrics = BigGlyphMetrics()
        dummy, data = sstruct.unpack2(bigGlyphMetricsFormat, self.data, self.metrics)
        dataLen, = struct.unpack('>L', data[:4])
        data = data[4:]
        assert dataLen <= len(data), 'Data overun in format 18'
        self.imageData = data[:dataLen]

    def compile(self, ttFont):
        dataList = []
        dataList.append(sstruct.pack(bigGlyphMetricsFormat, self.metrics))
        dataList.append(struct.pack('>L', len(self.imageData)))
        dataList.append(self.imageData)
        return bytesjoin(dataList)