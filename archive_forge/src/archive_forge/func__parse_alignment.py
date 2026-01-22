import sys
import logging
import os
import re
from xml.etree.ElementTree import parse, SubElement
from .objcreator import QObjectCreator
from .properties import Properties
def _parse_alignment(alignment):
    """ Convert a C++ alignment to the corresponding flags. """
    align_flags = None
    for qt_align in alignment.split('|'):
        _, qt_align = qt_align.split('::')
        align = getattr(QtCore.Qt, qt_align)
        if align_flags is None:
            align_flags = align
        else:
            align_flags |= align
    return align_flags