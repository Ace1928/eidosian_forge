from fontTools.ttLib import TTFont, TTLibError
from fontTools.misc.macCreatorType import getMacCreatorAndType
from fontTools.unicode import setUnicodeData
from fontTools.misc.textTools import Tag, tostr
from fontTools.misc.timeTools import timestampSinceEpoch
from fontTools.misc.loggingTools import Timer
from fontTools.misc.cliTools import makeOutputFileName
import os
import sys
import getopt
import re
import logging
def ttList(input, output, options):
    ttf = TTFont(input, fontNumber=options.fontNumber, lazy=True)
    reader = ttf.reader
    tags = sorted(reader.keys())
    print('Listing table info for "%s":' % input)
    format = '    %4s  %10s  %8s  %8s'
    print(format % ('tag ', '  checksum', '  length', '  offset'))
    print(format % ('----', '----------', '--------', '--------'))
    for tag in tags:
        entry = reader.tables[tag]
        if ttf.flavor == 'woff2':
            from fontTools.ttLib.sfnt import calcChecksum
            data = entry.loadData(reader.transformBuffer)
            checkSum = calcChecksum(data)
        else:
            checkSum = int(entry.checkSum)
        if checkSum < 0:
            checkSum = checkSum + 4294967296
        checksum = '0x%08X' % checkSum
        print(format % (tag, checksum, entry.length, entry.offset))
    print()
    ttf.close()