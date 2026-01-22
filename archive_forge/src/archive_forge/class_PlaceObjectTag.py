import os
import zlib
import time  # noqa
import logging
import numpy as np
class PlaceObjectTag(ControlTag):

    def __init__(self, depth, idToPlace=None, xy=(0, 0), move=False):
        ControlTag.__init__(self)
        self.tagtype = 26
        self.depth = depth
        self.idToPlace = idToPlace
        self.xy = xy
        self.move = move

    def process_tag(self):
        depth = self.depth
        xy = self.xy
        id = self.idToPlace
        bb = bytes()
        if self.move:
            bb += '\x07'.encode('ascii')
        else:
            bb += '\x06'.encode('ascii')
        bb += int2uint16(depth)
        bb += int2uint16(id)
        bb += self.make_matrix_record(trans_xy=xy).tobytes()
        self.bytes = bb