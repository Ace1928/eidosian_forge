import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def _DoRemapping(element, map):
    """If |element| then remap it through |map|. If |element| is iterable then
    each item will be remapped. Any elements not found will be removed."""
    if map is not None and element is not None:
        if not callable(map):
            map = map.get
        if isinstance(element, list) or isinstance(element, tuple):
            element = filter(None, [map(elem) for elem in element])
        else:
            element = map(element)
    return element