import os
import subprocess
import tempfile
import warnings
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB import Selection
from Bio.PDB.AbstractPropertyMap import AbstractPropertyMap
from Bio.PDB.Polypeptide import is_aa
from Bio import BiopythonWarning
def _get_atom_radius(atom, rtype='united'):
    """Translate an atom object to an atomic radius defined in MSMS (PRIVATE).

    Uses information from the parent residue and the atom object to define
    the atom type.

    Returns the radius (float) according to the selected type:
     - explicit (reads hydrogens)
     - united (default)

    """
    if rtype == 'explicit':
        typekey = 1
    elif rtype == 'united':
        typekey = 2
    else:
        raise ValueError(f"Radius type ({rtype!r}) not understood. Must be 'explicit' or 'united'")
    resname = atom.parent.resname
    het_atm = atom.parent.id[0]
    at_name = atom.name
    at_elem = atom.element
    if at_elem == 'H' or at_elem == 'D':
        return _atomic_radii[15][typekey]
    elif het_atm == 'W' and at_elem == 'O':
        return _atomic_radii[2][typekey]
    elif het_atm != ' ' and at_elem == 'CA':
        return _atomic_radii[18][typekey]
    elif het_atm != ' ' and at_elem == 'CD':
        return _atomic_radii[22][typekey]
    elif resname == 'ACE' and at_name == 'CA':
        return _atomic_radii[9][typekey]
    elif at_name == 'N':
        return _atomic_radii[4][typekey]
    elif at_name == 'CA':
        return _atomic_radii[7][typekey]
    elif at_name == 'C':
        return _atomic_radii[10][typekey]
    elif at_name == 'O':
        return _atomic_radii[1][typekey]
    elif at_name == 'P':
        return _atomic_radii[13][typekey]
    elif at_name == 'CB' and resname == 'ALA':
        return _atomic_radii[9][typekey]
    elif at_name == 'CB' and resname in {'ILE', 'THR', 'VAL'}:
        return _atomic_radii[7][typekey]
    elif at_name == 'CB':
        return _atomic_radii[8][typekey]
    elif at_name == 'CG' and resname in {'ASN', 'ASP', 'ASX', 'HIS', 'HIP', 'HIE', 'HID', 'HISN', 'HISL', 'LEU', 'PHE', 'TRP', 'TYR'}:
        return _atomic_radii[10][typekey]
    elif at_name == 'CG' and resname == 'LEU':
        return _atomic_radii[7][typekey]
    elif at_name == 'CG':
        return _atomic_radii[8][typekey]
    elif resname == 'GLN' and at_elem == 'O':
        return _atomic_radii[3][typekey]
    elif resname == 'ACE' and at_name == 'CH3':
        return _atomic_radii[9][typekey]
    elif resname == 'ARG' and at_name == 'CD':
        return _atomic_radii[8][typekey]
    elif resname == 'ARG' and at_name in {'NE', 'RE'}:
        return _atomic_radii[4][typekey]
    elif resname == 'ARG' and at_name == 'CZ':
        return _atomic_radii[10][typekey]
    elif resname == 'ARG' and at_name.startswith(('NH', 'RH')):
        return _atomic_radii[5][typekey]
    elif resname == 'ASN' and at_name == 'OD1':
        return _atomic_radii[1][typekey]
    elif resname == 'ASN' and at_name == 'ND2':
        return _atomic_radii[5][typekey]
    elif resname == 'ASN' and at_name.startswith('AD'):
        return _atomic_radii[3][typekey]
    elif resname == 'ASP' and at_name.startswith(('OD', 'ED')):
        return _atomic_radii[3][typekey]
    elif resname == 'ASX' and at_name.startswith('OD1'):
        return _atomic_radii[1][typekey]
    elif resname == 'ASX' and at_name == 'ND2':
        return _atomic_radii[3][typekey]
    elif resname == 'ASX' and at_name.startswith(('OD', 'AD')):
        return _atomic_radii[3][typekey]
    elif resname in {'CYS', 'CYX', 'CYM'} and at_name == 'SG':
        return _atomic_radii[13][typekey]
    elif resname in {'CYS', 'MET'} and at_name.startswith('LP'):
        return _atomic_radii[13][typekey]
    elif resname == 'CUH' and at_name == 'SG':
        return _atomic_radii[12][typekey]
    elif resname == 'GLU' and at_name.startswith(('OE', 'EE')):
        return _atomic_radii[3][typekey]
    elif resname in {'GLU', 'GLN', 'GLX'} and at_name == 'CD':
        return _atomic_radii[10][typekey]
    elif resname == 'GLN' and at_name == 'OE1':
        return _atomic_radii[1][typekey]
    elif resname == 'GLN' and at_name == 'NE2':
        return _atomic_radii[5][typekey]
    elif resname in {'GLN', 'GLX'} and at_name.startswith('AE'):
        return _atomic_radii[3][typekey]
    elif resname in {'HIS', 'HID', 'HIE', 'HIP', 'HISL'} and at_name in {'CE1', 'CD2'}:
        return _atomic_radii[11][typekey]
    elif resname in {'HIS', 'HID', 'HIE', 'HISL'} and at_name == 'ND1':
        return _atomic_radii[14][typekey]
    elif resname in {'HID', 'HIP'} and at_name in {'ND1', 'RD1'}:
        return _atomic_radii[4][typekey]
    elif resname in {'HIS', 'HIE', 'HIP'} and at_name in {'NE2', 'RE2'}:
        return _atomic_radii[4][typekey]
    elif resname in {'HID', 'HISL'} and at_name in {'NE2', 'RE2'}:
        return _atomic_radii[14][typekey]
    elif resname in {'HIS', 'HID', 'HIP', 'HISL'} and at_name.startswith(('AD', 'AE')):
        return _atomic_radii[4][typekey]
    elif resname == 'ILE' and at_name == 'CG1':
        return _atomic_radii[8][typekey]
    elif resname == 'ILE' and at_name == 'CG2':
        return _atomic_radii[9][typekey]
    elif resname == 'ILE' and at_name in {'CD', 'CD1'}:
        return _atomic_radii[9][typekey]
    elif resname == 'LEU' and at_name.startswith('CD'):
        return _atomic_radii[9][typekey]
    elif resname == 'LYS' and at_name in {'CG', 'CD', 'CE'}:
        return _atomic_radii[8][typekey]
    elif resname == 'LYS' and at_name in {'NZ', 'KZ'}:
        return _atomic_radii[6][typekey]
    elif resname == 'MET' and at_name == 'SD':
        return _atomic_radii[13][typekey]
    elif resname == 'MET' and at_name == 'CE':
        return _atomic_radii[9][typekey]
    elif resname == 'PHE' and at_name.startswith(('CD', 'CE', 'CZ')):
        return _atomic_radii[11][typekey]
    elif resname == 'PRO' and at_name in {'CG', 'CD'}:
        return _atomic_radii[8][typekey]
    elif resname == 'CSO' and at_name in {'SE', 'SEG'}:
        return _atomic_radii[9][typekey]
    elif resname == 'CSO' and at_name.startswith('OD'):
        return _atomic_radii[3][typekey]
    elif resname == 'SER' and at_name == 'OG':
        return _atomic_radii[2][typekey]
    elif resname == 'THR' and at_name == 'OG1':
        return _atomic_radii[2][typekey]
    elif resname == 'THR' and at_name == 'CG2':
        return _atomic_radii[9][typekey]
    elif resname == 'TRP' and at_name == 'CD1':
        return _atomic_radii[11][typekey]
    elif resname == 'TRP' and at_name in {'CD2', 'CE2'}:
        return _atomic_radii[10][typekey]
    elif resname == 'TRP' and at_name == 'NE1':
        return _atomic_radii[4][typekey]
    elif resname == 'TRP' and at_name in {'CE3', 'CZ2', 'CZ3', 'CH2'}:
        return _atomic_radii[11][typekey]
    elif resname == 'TYR' and at_name in {'CD1', 'CD2', 'CE1', 'CE2'}:
        return _atomic_radii[11][typekey]
    elif resname == 'TYR' and at_name == 'CZ':
        return _atomic_radii[10][typekey]
    elif resname == 'TYR' and at_name == 'OH':
        return _atomic_radii[2][typekey]
    elif resname == 'VAL' and at_name in {'CG1', 'CG2'}:
        return _atomic_radii[9][typekey]
    elif at_name == 'CD':
        return _atomic_radii[8][typekey]
    elif resname in {'FS3', 'FS4'} and at_name.startswith('FE') and at_name.endswith(('1', '2', '3', '4', '5', '6', '7')):
        return _atomic_radii[21][typekey]
    elif resname in {'FS3', 'FS4'} and at_name.startswith('S') and at_name.endswith(('1', '2', '3', '4', '5', '6', '7')):
        return _atomic_radii[13][typekey]
    elif resname == 'FS3' and at_name == 'OXO':
        return _atomic_radii[1][typekey]
    elif resname == 'FEO' and at_name in {'FE1', 'FE2'}:
        return _atomic_radii[21][typekey]
    elif resname == 'HEM' and at_name in {'O1', 'O2'}:
        return _atomic_radii[1][typekey]
    elif resname == 'HEM' and at_name == 'FE':
        return _atomic_radii[21][typekey]
    elif resname == 'HEM' and at_name in {'CHA', 'CHB', 'CHC', 'CHD', 'CAB', 'CAC', 'CBB', 'CBC'}:
        return _atomic_radii[11][typekey]
    elif resname == 'HEM' and at_name in {'NA', 'NB', 'NC', 'ND', 'N A', 'N B', 'N C', 'N D'}:
        return _atomic_radii[14][typekey]
    elif resname == 'HEM' and at_name in {'C1A', 'C1B', 'C1C', 'C1D', 'C2A', 'C2B', 'C2C', 'C2D', 'C3A', 'C3B', 'C3C', 'C3D', 'C4A', 'C4B', 'C4C', 'C4D', 'CGA', 'CGD'}:
        return _atomic_radii[10][typekey]
    elif resname == 'HEM' and at_name in {'CMA', 'CMB', 'CMC', 'CMD'}:
        return _atomic_radii[9][typekey]
    elif resname == 'HEM' and at_name == 'OH2':
        return _atomic_radii[2][typekey]
    elif resname == 'AZI' and at_name in {'N1', 'N2', 'N3'}:
        return _atomic_radii[14][typekey]
    elif resname == 'MPD' and at_name in {'C1', 'C5', 'C6'}:
        return _atomic_radii[9][typekey]
    elif resname == 'MPD' and at_name == 'C2':
        return _atomic_radii[10][typekey]
    elif resname == 'MPD' and at_name == 'C3':
        return _atomic_radii[8][typekey]
    elif resname == 'MPD' and at_name == 'C4':
        return _atomic_radii[7][typekey]
    elif resname == 'MPD' and at_name in {'O7', 'O8'}:
        return _atomic_radii[2][typekey]
    elif resname in {'SO4', 'SUL'} and at_name == 'S':
        return _atomic_radii[13][typekey]
    elif resname in {'SO4', 'SUL', 'PO4', 'PHO'} and at_name in {'O1', 'O2', 'O3', 'O4'}:
        return _atomic_radii[3][typekey]
    elif resname == 'PC ' and at_name in {'O1', 'O2', 'O3', 'O4'}:
        return _atomic_radii[3][typekey]
    elif resname == 'PC ' and at_name == 'P1':
        return _atomic_radii[13][typekey]
    elif resname == 'PC ' and at_name in {'C1', 'C2'}:
        return _atomic_radii[8][typekey]
    elif resname == 'PC ' and at_name in {'C3', 'C4', 'C5'}:
        return _atomic_radii[9][typekey]
    elif resname == 'PC ' and at_name == 'N1':
        return _atomic_radii[14][typekey]
    elif resname == 'BIG' and at_name == 'BAL':
        return _atomic_radii[17][typekey]
    elif resname in {'POI', 'DOT'} and at_name in {'POI', 'DOT'}:
        return _atomic_radii[23][typekey]
    elif resname == 'FMN' and at_name in {'N1', 'N5', 'N10'}:
        return _atomic_radii[4][typekey]
    elif resname == 'FMN' and at_name in {'C2', 'C4', 'C7', 'C8', 'C10', 'C4A', 'C5A', 'C9A'}:
        return _atomic_radii[10][typekey]
    elif resname == 'FMN' and at_name in {'O2', 'O4'}:
        return _atomic_radii[1][typekey]
    elif resname == 'FMN' and at_name == 'N3':
        return _atomic_radii[14][typekey]
    elif resname == 'FMN' and at_name in {'C6', 'C9'}:
        return _atomic_radii[11][typekey]
    elif resname == 'FMN' and at_name in {'C7M', 'C8M'}:
        return _atomic_radii[9][typekey]
    elif resname == 'FMN' and at_name.startswith(('C1', 'C2', 'C3', 'C4', 'C5')):
        return _atomic_radii[8][typekey]
    elif resname == 'FMN' and at_name.startswith(('O2', 'O3', 'O4')):
        return _atomic_radii[2][typekey]
    elif resname == 'FMN' and at_name.startswith('O5'):
        return _atomic_radii[3][typekey]
    elif resname == 'FMN' and at_name in {'OP1', 'OP2', 'OP3'}:
        return _atomic_radii[3][typekey]
    elif resname in {'ALK', 'MYR'} and at_name == 'OT1':
        return _atomic_radii[3][typekey]
    elif resname in {'ALK', 'MYR'} and at_name == 'C01':
        return _atomic_radii[10][typekey]
    elif resname == 'ALK' and at_name == 'C16':
        return _atomic_radii[9][typekey]
    elif resname == 'MYR' and at_name == 'C14':
        return _atomic_radii[9][typekey]
    elif resname in {'ALK', 'MYR'} and at_name.startswith('C'):
        return _atomic_radii[8][typekey]
    elif at_elem == 'CU':
        return _atomic_radii[20][typekey]
    elif at_elem == 'ZN':
        return _atomic_radii[19][typekey]
    elif at_elem == 'MN':
        return _atomic_radii[27][typekey]
    elif at_elem == 'FE':
        return _atomic_radii[25][typekey]
    elif at_elem == 'MG':
        return _atomic_radii[26][typekey]
    elif at_elem == 'CO':
        return _atomic_radii[28][typekey]
    elif at_elem == 'SE':
        return _atomic_radii[29][typekey]
    elif at_elem == 'YB':
        return _atomic_radii[31][typekey]
    elif at_name == 'SEG':
        return _atomic_radii[9][typekey]
    elif at_name == 'OXT':
        return _atomic_radii[3][typekey]
    elif at_name.startswith(('OT', 'E')):
        return _atomic_radii[3][typekey]
    elif at_name.startswith('S'):
        return _atomic_radii[13][typekey]
    elif at_name.startswith('C'):
        return _atomic_radii[7][typekey]
    elif at_name.startswith('A'):
        return _atomic_radii[11][typekey]
    elif at_name.startswith('O'):
        return _atomic_radii[1][typekey]
    elif at_name.startswith(('N', 'R')):
        return _atomic_radii[4][typekey]
    elif at_name.startswith('K'):
        return _atomic_radii[6][typekey]
    elif at_name in {'PA', 'PB', 'PC', 'PD'}:
        return _atomic_radii[13][typekey]
    elif at_name.startswith('P'):
        return _atomic_radii[13][typekey]
    elif resname in {'FAD', 'NAD', 'AMX', 'APU'} and at_name.startswith('O'):
        return _atomic_radii[1][typekey]
    elif resname in {'FAD', 'NAD', 'AMX', 'APU'} and at_name.startswith('N'):
        return _atomic_radii[4][typekey]
    elif resname in {'FAD', 'NAD', 'AMX', 'APU'} and at_name.startswith('C'):
        return _atomic_radii[7][typekey]
    elif resname in {'FAD', 'NAD', 'AMX', 'APU'} and at_name.startswith('P'):
        return _atomic_radii[13][typekey]
    elif resname in {'FAD', 'NAD', 'AMX', 'APU'} and at_name.startswith('H'):
        return _atomic_radii[15][typekey]
    else:
        warnings.warn(f'{at_name}:{resname} not in radii library.', BiopythonWarning)
        return 0.01