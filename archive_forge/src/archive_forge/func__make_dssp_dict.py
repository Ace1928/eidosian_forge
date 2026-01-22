import re
import os
from io import StringIO
import subprocess
import warnings
from Bio.PDB.AbstractPropertyMap import AbstractResiduePropertyMap
from Bio.PDB.PDBExceptions import PDBException
from Bio.PDB.PDBParser import PDBParser
from Bio.Data.PDBData import protein_letters_3to1, residue_sasa_scales
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
def _make_dssp_dict(handle):
    """Return a DSSP dictionary, used by mask_dssp_dict (PRIVATE).

    DSSP dictionary maps (chainid, resid) to an amino acid,
    secondary structure symbol, solvent accessibility value, and hydrogen bond
    information (relative dssp indices and hydrogen bond energies) from an open
    DSSP file object.

    Parameters
    ----------
    handle : file
        the open DSSP output file handle

    """
    dssp = {}
    start = 0
    keys = []
    for line in handle:
        sl = line.split()
        if len(sl) < 2:
            continue
        if sl[1] == 'RESIDUE':
            start = 1
            continue
        if not start:
            continue
        if line[9] == ' ':
            continue
        dssp_index = int(line[:5])
        resseq = int(line[5:10])
        icode = line[10]
        chainid = line[11]
        aa = line[13]
        ss = line[16]
        if ss == ' ':
            ss = '-'
        try:
            NH_O_1_relidx = int(line[38:45])
            NH_O_1_energy = float(line[46:50])
            O_NH_1_relidx = int(line[50:56])
            O_NH_1_energy = float(line[57:61])
            NH_O_2_relidx = int(line[61:67])
            NH_O_2_energy = float(line[68:72])
            O_NH_2_relidx = int(line[72:78])
            O_NH_2_energy = float(line[79:83])
            acc = int(line[34:38])
            phi = float(line[103:109])
            psi = float(line[109:115])
        except ValueError as exc:
            if line[34] != ' ':
                shift = line[34:].find(' ')
                NH_O_1_relidx = int(line[38 + shift:45 + shift])
                NH_O_1_energy = float(line[46 + shift:50 + shift])
                O_NH_1_relidx = int(line[50 + shift:56 + shift])
                O_NH_1_energy = float(line[57 + shift:61 + shift])
                NH_O_2_relidx = int(line[61 + shift:67 + shift])
                NH_O_2_energy = float(line[68 + shift:72 + shift])
                O_NH_2_relidx = int(line[72 + shift:78 + shift])
                O_NH_2_energy = float(line[79 + shift:83 + shift])
                acc = int(line[34 + shift:38 + shift])
                phi = float(line[103 + shift:109 + shift])
                psi = float(line[109 + shift:115 + shift])
            else:
                raise ValueError(exc) from None
        res_id = (' ', resseq, icode)
        dssp[chainid, res_id] = (aa, ss, acc, phi, psi, dssp_index, NH_O_1_relidx, NH_O_1_energy, O_NH_1_relidx, O_NH_1_energy, NH_O_2_relidx, NH_O_2_energy, O_NH_2_relidx, O_NH_2_energy)
        keys.append((chainid, res_id))
    return (dssp, keys)