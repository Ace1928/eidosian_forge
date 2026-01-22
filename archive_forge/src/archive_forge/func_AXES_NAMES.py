from __future__ import annotations
import binascii
import collections
import datetime
import enum
import glob
import io
import json
import logging
import math
import os
import re
import struct
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
import numpy
from typing import TYPE_CHECKING, BinaryIO, cast, final, overload
@cached_property
def AXES_NAMES(self) -> dict[str, str]:
    """Map axes character codes to dimension names.

        - **X : width** (image width)
        - **Y : height** (image length)
        - **Z : depth** (image depth)
        - **S : sample** (color space and extra samples)
        - **I : sequence** (generic sequence of images, frames, planes, pages)
        - **T : time** (time series)
        - **C : channel** (acquisition path or emission wavelength)
        - **A : angle** (OME)
        - **P : phase** (OME. In LSM, **P** maps to **position**)
        - **R : tile** (OME. Region, position, or mosaic)
        - **H : lifetime** (OME. Histogram)
        - **E : lambda** (OME. Excitation wavelength)
        - **Q : other** (OME)
        - **L : exposure** (FluoView)
        - **V : event** (FluoView)
        - **M : mosaic** (LSM 6)
        - **J : column** (NDTiff)
        - **K : row** (NDTiff)

        There is no universal standard for dimension codes or names.
        This mapping mainly follows TIFF, OME-TIFF, ImageJ, LSM, and FluoView
        conventions.

        """
    return {'X': 'width', 'Y': 'height', 'Z': 'depth', 'S': 'sample', 'I': 'sequence', 'T': 'time', 'C': 'channel', 'A': 'angle', 'P': 'phase', 'R': 'tile', 'H': 'lifetime', 'E': 'lambda', 'L': 'exposure', 'V': 'event', 'M': 'mosaic', 'Q': 'other', 'J': 'column', 'K': 'row'}