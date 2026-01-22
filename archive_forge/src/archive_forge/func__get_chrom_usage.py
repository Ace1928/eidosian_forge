import sys
import io
import copy
import array
import itertools
import struct
import zlib
from collections import namedtuple
from io import BytesIO
import numpy as np
from Bio.Align import Alignment
from Bio.Align import interfaces
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
def _get_chrom_usage(cls, alignments, targets, extra_indices):
    aveSize = 0
    chromId = 0
    totalBases = 0
    bedCount = 0
    name = ''
    chromUsageList = []
    keySize = 0
    chromSize = -1
    minDiff = sys.maxsize
    for alignment in alignments:
        chrom = alignment.sequences[0].id
        start = alignment.coordinates[0, 0]
        end = alignment.coordinates[0, -1]
        for extra_index in extra_indices:
            extra_index.updateMaxFieldSize(alignment)
        if start > end:
            raise ValueError(f'end ({end}) before start ({start}) in alignment [{bedCount}]')
        bedCount += 1
        totalBases += end - start
        if name != chrom:
            if name > chrom:
                raise ValueError(f'alignments are not sorted by target name at alignment [{bedCount}]')
            if name:
                chromUsageList.append((name, chromId, chromSize))
                chromId += 1
            for target in targets:
                if target.id == chrom:
                    break
            else:
                raise ValueError(f"failed to find target '{chrom}' in target list at alignment [{bedCount}]")
            name = chrom
            keySize = max(keySize, len(chrom))
            chromSize = len(target)
            lastStart = -1
        if end > chromSize:
            raise ValueError(f"end coordinate {end} bigger than {chrom} size of {chromSize} at alignment [{bedCount}]'")
        if lastStart >= 0:
            diff = start - lastStart
            if diff < minDiff:
                if diff < 0:
                    raise ValueError(f'alignments are not sorted at alignment [{bedCount}]')
                minDiff = diff
        lastStart = start
    if name:
        chromUsageList.append((name, chromId, chromSize))
    chromUsageList = np.array(chromUsageList, dtype=[('name', f'S{keySize}'), ('id', '=i4'), ('size', '=i4')])
    if bedCount > 0:
        aveSize = totalBases / bedCount
    alignments._len = bedCount
    return (chromUsageList, aveSize)