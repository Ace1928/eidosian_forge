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
class _RTreeFormatter:
    signature = 610839776

    def __init__(self, byteorder='='):
        self.formatter_header = struct.Struct(byteorder + 'IIQIIIIQIxxxx')
        self.formatter_node = struct.Struct(byteorder + '?xH')
        self.formatter_nonleaf = struct.Struct(byteorder + 'IIIIQ')
        self.formatter_leaf = struct.Struct(byteorder + 'IIIIQQ')

    def read(self, stream):
        NonLeaf = namedtuple('NonLeaf', ['parent', 'children', 'startChromIx', 'startBase', 'endChromIx', 'endBase', 'dataOffset'])
        Leaf = namedtuple('Leaf', ['parent', 'startChromIx', 'startBase', 'endChromIx', 'endBase', 'dataOffset', 'dataSize'])
        data = stream.read(self.formatter_header.size)
        magic, blockSize, itemCount, startChromIx, startBase, endChromIx, endBase, endFileOffset, itemsPerSlot = self.formatter_header.unpack(data)
        assert magic == _RTreeFormatter.signature
        formatter_node = self.formatter_node
        formatter_nonleaf = self.formatter_nonleaf
        formatter_leaf = self.formatter_leaf
        root = NonLeaf(None, [], startChromIx, startBase, endChromIx, endBase, None)
        node = root
        itemsCounted = 0
        while True:
            data = stream.read(formatter_node.size)
            isLeaf, count = formatter_node.unpack(data)
            if isLeaf:
                children = node.children
                for i in range(count):
                    data = stream.read(formatter_leaf.size)
                    startChromIx, startBase, endChromIx, endBase, dataOffset, dataSize = formatter_leaf.unpack(data)
                    child = Leaf(node, startChromIx, startBase, endChromIx, endBase, dataOffset, dataSize)
                    children.append(child)
                itemsCounted += count
                while True:
                    parent = node.parent
                    if parent is None:
                        assert itemsCounted == itemCount
                        return node
                    for index, child in enumerate(parent.children):
                        if id(node) == id(child):
                            break
                    else:
                        raise RuntimeError('Failed to find child node')
                    try:
                        node = parent.children[index + 1]
                    except IndexError:
                        node = parent
                    else:
                        break
            else:
                children = node.children
                for i in range(count):
                    data = stream.read(formatter_nonleaf.size)
                    startChromIx, startBase, endChromIx, endBase, dataOffset = formatter_nonleaf.unpack(data)
                    child = NonLeaf(node, [], startChromIx, startBase, endChromIx, endBase, dataOffset)
                    children.append(child)
                parent = node
                node = children[0]
            stream.seek(node.dataOffset)

    def rTreeFromChromRangeArray(self, blockSize, items, endFileOffset):
        itemCount = len(items)
        if itemCount == 0:
            return
        children = []
        nextOffset = items[0].offset
        oneSize = 0
        i = 0
        while i < itemCount:
            child = _RTreeNode()
            children.append(child)
            startItem = items[i]
            child.startChromId = child.endChromId = startItem.chromId
            child.startBase = startItem.start
            child.endBase = startItem.end
            child.startFileOffset = nextOffset
            oneSize = 1
            endItem = startItem
            for j in range(i + 1, itemCount):
                endItem = items[j]
                nextOffset = endItem.offset
                if nextOffset != child.startFileOffset:
                    break
                oneSize += 1
            else:
                nextOffset = endFileOffset
            child.endFileOffset = nextOffset
            for item in items[i + 1:i + oneSize]:
                if item.chromId < child.startChromId:
                    child.startChromId = item.chromId
                    child.startBase = item.start
                elif item.chromId == child.startChromId and item.start < child.startBase:
                    child.startBase = item.start
                if item.chromId > child.endChromId:
                    child.endChromId = item.chromId
                    child.endBase = item.end
                elif item.chromId == child.endChromId and item.end > child.endBase:
                    child.endBase = item.end
            i += oneSize
        levelCount = 1
        while True:
            parents = []
            slotsUsed = blockSize
            for child in children:
                if slotsUsed >= blockSize:
                    slotsUsed = 1
                    parent = _RTreeNode()
                    parent.parent = child.parent
                    parent.startChromId = child.startChromId
                    parent.startBase = child.startBase
                    parent.endChromId = child.endChromId
                    parent.endBase = child.endBase
                    parent.startFileOffset = child.startFileOffset
                    parent.endFileOffset = child.endFileOffset
                    parents.append(parent)
                else:
                    slotsUsed += 1
                    if child.startChromId < parent.startChromId:
                        parent.startChromId = child.startChromId
                        parent.startBase = child.startBase
                    elif child.startChromId == parent.startChromId and child.startBase < parent.startBase:
                        parent.startBase = child.startBase
                    if child.endChromId > parent.endChromId:
                        parent.endChromId = child.endChromId
                        parent.endBase = child.endBase
                    elif child.endChromId == parent.endChromId and child.endBase > parent.endBase:
                        parent.endBase = child.endBase
                parent.children.append(child)
                child.parent = parent
            levelCount += 1
            if len(parents) == 1:
                break
            children = parents
        return (parent, levelCount)

    def rWriteLeaves(self, itemsPerSlot, lNodeSize, tree, curLevel, leafLevel, output):
        formatter_leaf = self.formatter_leaf
        if curLevel == leafLevel:
            isLeaf = True
            data = self.formatter_node.pack(isLeaf, len(tree.children))
            output.write(data)
            for child in tree.children:
                data = formatter_leaf.pack(child.startChromId, child.startBase, child.endChromId, child.endBase, child.startFileOffset, child.endFileOffset - child.startFileOffset)
                output.write(data)
            output.write(bytes((itemsPerSlot - len(tree.children)) * self.formatter_nonleaf.size))
        else:
            for child in tree.children:
                self.rWriteLeaves(itemsPerSlot, lNodeSize, child, curLevel + 1, leafLevel, output)

    def rWriteIndexLevel(self, parent, blockSize, childNodeSize, curLevel, destLevel, offset, output):
        previous_offset = offset
        formatter_nonleaf = self.formatter_nonleaf
        if curLevel == destLevel:
            isLeaf = False
            data = self.formatter_node.pack(isLeaf, len(parent.children))
            output.write(data)
            for child in parent.children:
                data = formatter_nonleaf.pack(child.startChromId, child.startBase, child.endChromId, child.endBase, offset)
                output.write(data)
                offset += childNodeSize
            output.write(bytes((blockSize - len(parent.children)) * self.formatter_nonleaf.size))
        else:
            for child in parent.children:
                offset = self.rWriteIndexLevel(child, blockSize, childNodeSize, curLevel + 1, destLevel, offset, output)
        position = output.tell()
        if position != previous_offset:
            raise RuntimeError(f'Internal error: offset mismatch ({position} vs {previous_offset})')
        return offset

    def write(self, items, blockSize, itemsPerSlot, endFileOffset, output):
        root, levelCount = self.rTreeFromChromRangeArray(blockSize, items, endFileOffset)
        data = self.formatter_header.pack(_RTreeFormatter.signature, blockSize, len(items), root.startChromId, root.startBase, root.endChromId, root.endBase, endFileOffset, itemsPerSlot)
        output.write(data)
        if root is None:
            return
        levelSizes = np.zeros(levelCount, int)
        root.calcLevelSizes(levelSizes, level=0)
        size = self.formatter_node.size + self.formatter_nonleaf.size * blockSize
        levelOffset = output.tell()
        for i in range(levelCount - 2):
            levelOffset += levelSizes[i] * size
            if i == levelCount - 3:
                size = self.formatter_node.size + self.formatter_leaf.size * blockSize
            self.rWriteIndexLevel(root, blockSize, size, 0, i, levelOffset, output)
        leafLevel = levelCount - 2
        self.rWriteLeaves(blockSize, size, root, 0, leafLevel, output)