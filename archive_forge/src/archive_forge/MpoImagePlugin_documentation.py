from __future__ import annotations
import itertools
import os
import struct
from . import (
from ._binary import i16be as i16
from ._binary import o32le

        Transform the instance of JpegImageFile into
        an instance of MpoImageFile.
        After the call, the JpegImageFile is extended
        to be an MpoImageFile.

        This is essentially useful when opening a JPEG
        file that reveals itself as an MPO, to avoid
        double call to _open.
        