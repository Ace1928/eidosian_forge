import xcffib
import struct
import io
class ConfigWindow:
    X = 1 << 0
    Y = 1 << 1
    Width = 1 << 2
    Height = 1 << 3
    BorderWidth = 1 << 4
    Sibling = 1 << 5
    StackMode = 1 << 6