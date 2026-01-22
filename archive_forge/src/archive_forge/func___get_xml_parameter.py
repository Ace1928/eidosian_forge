import re
import numpy as np
from ase import Atoms
from ase.utils import reader, writer
from ase.io.utils import ImageIterator
from ase.io import ParseError
from .vasp_parsers import vasp_outcar_parsers as vop
from pathlib import Path
def __get_xml_parameter(par):
    """An auxiliary function that enables convenient extraction of
    parameter values from a vasprun.xml file with proper type
    handling.

    """

    def to_bool(b):
        if b == 'T':
            return True
        else:
            return False
    to_type = {'int': int, 'logical': to_bool, 'string': str, 'float': float}
    text = par.text
    if text is None:
        text = ''
    var_type = to_type[par.attrib.get('type', 'float')]
    try:
        if par.tag == 'v':
            return list(map(var_type, text.split()))
        else:
            return var_type(text.strip())
    except ValueError:
        return None