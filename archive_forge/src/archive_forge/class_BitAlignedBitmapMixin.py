from fontTools.misc import sstruct
from fontTools.misc.textTools import (
from .BitmapGlyphMetrics import (
from . import DefaultTable
import itertools
import os
import struct
import logging
class BitAlignedBitmapMixin(object):

    def _getBitRange(self, row, bitDepth, metrics):
        rowBits = bitDepth * metrics.width
        bitOffset = row * rowBits
        return (bitOffset, bitOffset + rowBits)

    def getRow(self, row, bitDepth=1, metrics=None, reverseBytes=False):
        if metrics is None:
            metrics = self.metrics
        assert 0 <= row and row < metrics.height, 'Illegal row access in bitmap'
        dataList = []
        bitRange = self._getBitRange(row, bitDepth, metrics)
        stepRange = bitRange + (8,)
        for curBit in range(*stepRange):
            endBit = min(curBit + 8, bitRange[1])
            numBits = endBit - curBit
            cutPoint = curBit % 8
            firstByteLoc = curBit // 8
            secondByteLoc = endBit // 8
            if firstByteLoc < secondByteLoc:
                numBitsCut = 8 - cutPoint
            else:
                numBitsCut = endBit - curBit
            curByte = _reverseBytes(self.imageData[firstByteLoc])
            firstHalf = byteord(curByte) >> cutPoint
            firstHalf = (1 << numBitsCut) - 1 & firstHalf
            newByte = firstHalf
            if firstByteLoc < secondByteLoc and secondByteLoc < len(self.imageData):
                curByte = _reverseBytes(self.imageData[secondByteLoc])
                secondHalf = byteord(curByte) << numBitsCut
                newByte = (firstHalf | secondHalf) & (1 << numBits) - 1
            dataList.append(bytechr(newByte))
        data = bytesjoin(dataList)
        if not reverseBytes:
            data = _reverseBytes(data)
        return data

    def setRows(self, dataRows, bitDepth=1, metrics=None, reverseBytes=False):
        if metrics is None:
            metrics = self.metrics
        if not reverseBytes:
            dataRows = list(map(_reverseBytes, dataRows))
        numBytes = (self._getBitRange(len(dataRows), bitDepth, metrics)[0] + 7) // 8
        ordDataList = [0] * numBytes
        for row, data in enumerate(dataRows):
            bitRange = self._getBitRange(row, bitDepth, metrics)
            stepRange = bitRange + (8,)
            for curBit, curByte in zip(range(*stepRange), data):
                endBit = min(curBit + 8, bitRange[1])
                cutPoint = curBit % 8
                firstByteLoc = curBit // 8
                secondByteLoc = endBit // 8
                if firstByteLoc < secondByteLoc:
                    numBitsCut = 8 - cutPoint
                else:
                    numBitsCut = endBit - curBit
                curByte = byteord(curByte)
                firstByte = curByte & (1 << numBitsCut) - 1
                ordDataList[firstByteLoc] |= firstByte << cutPoint
                if firstByteLoc < secondByteLoc and secondByteLoc < numBytes:
                    secondByte = curByte >> numBitsCut & (1 << 8 - numBitsCut) - 1
                    ordDataList[secondByteLoc] |= secondByte
        self.imageData = _reverseBytes(bytesjoin(map(bytechr, ordDataList)))