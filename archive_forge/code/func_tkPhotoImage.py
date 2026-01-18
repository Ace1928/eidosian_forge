from __future__ import annotations
import os
import struct
import sys
from . import Image, ImageFile
def tkPhotoImage(self):
    from . import ImageTk
    return ImageTk.PhotoImage(self.convert2byte(), palette=256)