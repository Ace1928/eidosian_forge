import re
import time
import numpy as np
from ase.atoms import Atoms
from ase.utils import reader, writer
from ase.cell import Cell
def _read_process_rmc6f_lines_to_pos_and_cell(lines):
    """
    Processes the lines of rmc6f file to atom position dictionary and cell

    Parameters
    ----------
    lines: list[str]
        List of lines from rmc6f file.

    Returns
    ------
    pos : dict{int:list[str|float]}
        Dict for each atom id and Atoms properties based on rmc6f style.
        Basically, have 1) element and fractional coordinates for 'labels'
        or 'no_labels' style and 2) same for 'magnetic' style except adds
        the spin.
        Examples for 1) 'labels' or 'no_labels' styles or 2) 'magnetic' style:
            1) pos[aid] = [element, xf, yf, zf]
            2) pos[aid] = [element, xf, yf, zf, spin]
    cell: Cell object
        The ASE Cell object created from cell parameters read from the 'Cell'
        section of rmc6f file.
    """
    pos = {}
    header_lines = ['Number of atoms:', 'Supercell dimensions:', 'Cell (Ang/deg):', 'Lattice vectors (Ang):']
    sections = ['Atoms']
    header_lines_re = _read_construct_regex(header_lines)
    sections_re = _read_construct_regex(sections)
    section = None
    header = True
    lines = [line for line in lines if line != '']
    pos = {}
    for line in lines:
        m = re.match(sections_re, line)
        if m is not None:
            section = m.group(0).strip()
            header = False
            continue
        if header:
            field = None
            val = None
            float_list_re = '\\s+(\\d[\\d|\\s\\.]+[\\d|\\.])'
            m = re.search(header_lines_re + float_list_re, line)
            if m is not None:
                field = m.group(1)
                val = m.group(2)
            if field is not None and val is not None:
                if field == 'Number of atoms:':
                    pass
                    '\n                    NOTE: Can just capture via number of atoms ingested.\n                          Maybe use in future for a check.\n                    code: natoms = int(val)\n                    '
                if field.startswith('Supercell'):
                    pass
                    '\n                    NOTE: wrapping back down to unit cell is not\n                          necessarily needed for ASE object.\n\n                    code: supercell = [int(x) for x in val.split()]\n                    '
                if field.startswith('Cell'):
                    cellpar = [float(x) for x in val.split()]
                    cell = Cell.fromcellpar(cellpar)
                if field.startswith('Lattice'):
                    pass
                    '\n                    NOTE: Have questions about RMC fractionalization matrix for\n                          conversion in data2config vs. the ASE matrix.\n                          Currently, just support the Cell section.\n                    '
        if section is not None:
            if section == 'Atoms':
                atom_id, atom_props = _read_line_of_atoms_section(line.split())
                pos[atom_id] = atom_props
    return (pos, cell)