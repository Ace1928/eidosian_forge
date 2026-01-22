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
def _philips_load_pages(self) -> None:
    """Read and fix all pages from Philips slide file.

        The imagewidth and imagelength values of all tiled pages are corrected
        using the DICOM_PIXEL_SPACING attributes of the XML formatted
        description of the first page.

        """
    from xml.etree import ElementTree as etree
    pages = self.pages
    pages.cache = True
    pages.useframes = False
    pages._load()
    npages = len(pages)
    page0 = self.pages.first
    root = etree.fromstring(page0.description)
    width = float(page0.imagewidth)
    length = float(page0.imagelength)
    sizes: list[tuple[int, int]] | None = None
    for elem in root.iter():
        if elem.tag != 'Attribute' or elem.attrib.get('Name', '') != 'DICOM_PIXEL_SPACING' or elem.text is None:
            continue
        w, h = (float(v) for v in elem.text.replace('"', '').split())
        if sizes is None:
            length *= h
            width *= w
            sizes = []
        else:
            sizes.append((int(math.ceil(length / h)), int(math.ceil(width / w))))
    assert sizes is not None
    i = 0
    for imagelength, imagewidth in sizes:
        while i < npages and cast(TiffPage, pages[i]).tilewidth == 0:
            i += 1
            continue
        if i == npages:
            break
        page = pages[i]
        assert isinstance(page, TiffPage)
        page.imagewidth = imagewidth
        page.imagelength = imagelength
        if page.shaped[-1] > 1:
            page.shape = (imagelength, imagewidth, page.shape[-1])
        elif page.shaped[0] > 1:
            page.shape = (page.shape[0], imagelength, imagewidth)
        else:
            page.shape = (imagelength, imagewidth)
        page.shaped = page.shaped[:2] + (imagelength, imagewidth) + page.shaped[-1:]
        i += 1