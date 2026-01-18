import importlib
import logging
import re
from io import StringIO
from xml.dom import minidom
from xml.parsers.expat import ExpatError
from rdkit.Chem import Mol
def unpatchPandas():
    if orig_to_html:
        setattr(to_html_class, 'to_html', orig_to_html)
    if orig_write_cell:
        setattr(html_formatter_class, '_write_cell', orig_write_cell)
    if orig_get_adjustment:
        setattr(pandas_formats.format, get_adjustment_name, orig_get_adjustment)
    if orig_get_formatter:
        setattr(dataframeformatter_class, '_get_formatter', orig_get_formatter)