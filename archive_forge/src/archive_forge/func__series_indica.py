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
def _series_indica(self) -> list[TiffPageSeries] | None:
    """Return pyramidal image series in IndicaLabs file."""
    from xml.etree import ElementTree as etree
    series = self._series_generic()
    if series is None or len(series) != 1:
        return series
    try:
        tree = etree.fromstring(self.pages.first.description)
    except etree.ParseError as exc:
        logger().error(f'{self!r} Indica series raised {exc!r}')
        return series
    channel_names = [channel.attrib['name'] for channel in tree.iter('channel')]
    for s in series:
        s.kind = 'indica'
        if s.axes[0] == 'I' and s.shape[0] == len(channel_names):
            s._set_dimensions(s.shape, 'C' + s.axes[1:], None, True)
        if s.is_pyramidal:
            s.name = 'Baseline'
    self.is_uniform = False
    return series