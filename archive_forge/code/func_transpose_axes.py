from __future__ import division, print_function
import sys
import os
import io
import re
import glob
import math
import zlib
import time
import json
import enum
import struct
import pathlib
import warnings
import binascii
import tempfile
import datetime
import threading
import collections
import multiprocessing
import concurrent.futures
import numpy
def transpose_axes(image, axes, asaxes='CTZYX'):
    """Return image with its axes permuted to match specified axes.

    A view is returned if possible.

    >>> transpose_axes(numpy.zeros((2, 3, 4, 5)), 'TYXC', asaxes='CTZYX').shape
    (5, 2, 1, 3, 4)

    """
    for ax in axes:
        if ax not in asaxes:
            raise ValueError('unknown axis %s' % ax)
    shape = image.shape
    for ax in reversed(asaxes):
        if ax not in axes:
            axes = ax + axes
            shape = (1,) + shape
    image = image.reshape(shape)
    image = image.transpose([axes.index(ax) for ax in asaxes])
    return image