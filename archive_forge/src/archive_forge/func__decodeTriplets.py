from io import BytesIO
import sys
import array
import struct
from collections import OrderedDict
from fontTools.misc import sstruct
from fontTools.misc.arrayTools import calcIntBounds
from fontTools.misc.textTools import Tag, bytechr, byteord, bytesjoin, pad
from fontTools.ttLib import (
from fontTools.ttLib.sfnt import (
from fontTools.ttLib.tables import ttProgram, _g_l_y_f
import logging
def _decodeTriplets(self, glyph):

    def withSign(flag, baseval):
        assert 0 <= baseval and baseval < 65536, 'integer overflow'
        return baseval if flag & 1 else -baseval
    nPoints = glyph.endPtsOfContours[-1] + 1
    flagSize = nPoints
    if flagSize > len(self.flagStream):
        raise TTLibError("not enough 'flagStream' data")
    flagsData = self.flagStream[:flagSize]
    self.flagStream = self.flagStream[flagSize:]
    flags = array.array('B', flagsData)
    triplets = array.array('B', self.glyphStream)
    nTriplets = len(triplets)
    assert nPoints <= nTriplets
    x = 0
    y = 0
    glyph.coordinates = getTableModule('glyf').GlyphCoordinates.zeros(nPoints)
    glyph.flags = array.array('B')
    tripletIndex = 0
    for i in range(nPoints):
        flag = flags[i]
        onCurve = not bool(flag >> 7)
        flag &= 127
        if flag < 84:
            nBytes = 1
        elif flag < 120:
            nBytes = 2
        elif flag < 124:
            nBytes = 3
        else:
            nBytes = 4
        assert tripletIndex + nBytes <= nTriplets
        if flag < 10:
            dx = 0
            dy = withSign(flag, ((flag & 14) << 7) + triplets[tripletIndex])
        elif flag < 20:
            dx = withSign(flag, ((flag - 10 & 14) << 7) + triplets[tripletIndex])
            dy = 0
        elif flag < 84:
            b0 = flag - 20
            b1 = triplets[tripletIndex]
            dx = withSign(flag, 1 + (b0 & 48) + (b1 >> 4))
            dy = withSign(flag >> 1, 1 + ((b0 & 12) << 2) + (b1 & 15))
        elif flag < 120:
            b0 = flag - 84
            dx = withSign(flag, 1 + (b0 // 12 << 8) + triplets[tripletIndex])
            dy = withSign(flag >> 1, 1 + (b0 % 12 >> 2 << 8) + triplets[tripletIndex + 1])
        elif flag < 124:
            b2 = triplets[tripletIndex + 1]
            dx = withSign(flag, (triplets[tripletIndex] << 4) + (b2 >> 4))
            dy = withSign(flag >> 1, ((b2 & 15) << 8) + triplets[tripletIndex + 2])
        else:
            dx = withSign(flag, (triplets[tripletIndex] << 8) + triplets[tripletIndex + 1])
            dy = withSign(flag >> 1, (triplets[tripletIndex + 2] << 8) + triplets[tripletIndex + 3])
        tripletIndex += nBytes
        x += dx
        y += dy
        glyph.coordinates[i] = (x, y)
        glyph.flags.append(int(onCurve))
    bytesConsumed = tripletIndex
    self.glyphStream = self.glyphStream[bytesConsumed:]