import re
import time
import numpy as np
from ase.atoms import Atoms
from ase.utils import reader, writer
from ase.cell import Cell
def _read_line_of_atoms_section(fields):
    """
    Process `fields` line of Atoms section in rmc6f file and output extracted
    info as atom id (key) and list of properties for Atoms object (values).

    Parameters
    ----------
    fields: list[str]
        List of columns from line in rmc6f file.


    Returns
    ------
    atom_id: int
        Atom ID
    properties: list[str|float]
        List of Atoms properties based on rmc6f style.
        Basically, have 1) element and fractional coordinates for 'labels'
        or 'no_labels' style and 2) same for 'magnetic' style except adds
        the spin.
        Examples for 1) 'labels' or 'no_labels' styles or 2) 'magnetic' style:
            1) [element, xf, yf, zf]
            2) [element, xf, yf, zf, spin]
    """
    atom_id = int(fields[0])
    ncols = len(fields)
    style = ncols2style[ncols]
    properties = list()
    element = str(fields[1])
    if style == 'no_labels':
        xf = float(fields[2])
        yf = float(fields[3])
        zf = float(fields[4])
        properties = [element, xf, yf, zf]
    elif style == 'labels':
        xf = float(fields[3])
        yf = float(fields[4])
        zf = float(fields[5])
        properties = [element, xf, yf, zf]
    elif style == 'magnetic':
        xf = float(fields[2])
        yf = float(fields[3])
        zf = float(fields[4])
        spin = float(fields[10].strip('M:'))
        properties = [element, xf, yf, zf, spin]
    else:
        raise Exception('Unsupported style for parsing rmc6f file format.')
    return (atom_id, properties)