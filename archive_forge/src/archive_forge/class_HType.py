import xcffib
import struct
import io
class HType:
    FromServerTime = 1 << 0
    FromClientTime = 1 << 1
    FromClientSequence = 1 << 2