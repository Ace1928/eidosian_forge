import sys
import logging
import os
import re
from xml.etree.ElementTree import parse, SubElement
from .objcreator import QObjectCreator
from .properties import Properties
def _layout_position(elem):
    """ Return either (), (0, alignment), (row, column, rowspan, colspan) or
    (row, column, rowspan, colspan, alignment) depending on the type of layout
    and its configuration.  The result will be suitable to use as arguments to
    the layout.
    """
    row = elem.attrib.get('row')
    column = elem.attrib.get('column')
    alignment = elem.attrib.get('alignment')
    if row is None or column is None:
        if alignment is None:
            return ()
        return (0, _parse_alignment(alignment))
    row = int(row)
    column = int(column)
    rowspan = int(elem.attrib.get('rowspan', 1))
    colspan = int(elem.attrib.get('colspan', 1))
    if alignment is None:
        return (row, column, rowspan, colspan)
    return (row, column, rowspan, colspan, _parse_alignment(alignment))