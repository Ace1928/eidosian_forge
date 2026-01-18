import re
from datetime import date
from io import StringIO
import numpy as np
from Bio.File import as_handle
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB.parse_pdb_header import _parse_pdb_header_list
from Bio.PDB.PDBExceptions import PDBException
from Bio.Data.PDBData import protein_letters_1to3
from Bio.PDB.internal_coords import (
from Bio.PDB.ic_data import (
from typing import TextIO, Set, List, Tuple, Union, Optional
from Bio.PDB.Structure import Structure
from Bio.PDB.Residue import Residue
from Bio import SeqIO
def write_PIC(entity, file, pdbid=None, chainid=None, picFlags: int=IC_Residue.picFlagsDefault, hCut: Optional[Union[float, None]]=None, pCut: Optional[Union[float, None]]=None):
    """Write Protein Internal Coordinates (PIC) to file.

    See :func:`read_PIC` for file format.
    See :data:`IC_Residue.pic_accuracy` to vary numeric accuracy.
    Recurses to lower entity levels (M, C, R).

    :param Entity entity: Biopython PDB Entity object: S, M, C or R
    :param Bio.File file: :func:`.as_handle` file name or handle
    :param str pdbid: PDB idcode, read from entity if not supplied
    :param char chainid: PDB Chain ID, set from C level entity.id if needed
    :param int picFlags: boolean flags controlling output, defined in
        :data:`Bio.PDB.internal_coords.IC_Residue.pic_flags`

        * "psi",
        * "omg",
        * "phi",
        * "tau",  # tau hedron (N-Ca-C)
        * "chi1",
        * "chi2",
        * "chi3",
        * "chi4",
        * "chi5",
        * "pomg",  # proline omega
        * "chi",   # chi1 through chi5
        * "classic_b",  # psi | phi | tau | pomg
        * "classic",    # classic_b | chi
        * "hedra",      # all hedra including bond lengths
        * "primary",    # all primary dihedra
        * "secondary",  # all secondary dihedra (fixed angle from primary dihedra)
        * "all",        # hedra | primary | secondary
        * "initAtoms",  # XYZ coordinates of initial Tau (N-Ca-C)
        * "bFactors"

        default is everything::

            picFlagsDefault = (
                pic_flags.all | pic_flags.initAtoms | pic_flags.bFactors
            )

        Usage in your code::

            # just primary dihedra and all hedra
            picFlags = (
                IC_Residue.pic_flags.primary | IC_Residue.pic_flags.hedra
            )

            # no B-factors:
            picFlags = IC_Residue.picFlagsDefault
            picFlags &= ~IC_Residue.pic_flags.bFactors

        :func:`read_PIC` with `(defaults=True)` will use default values for
        anything left out

    :param float hCut: default None
        only write hedra with ref db angle std dev greater than this value
    :param float pCut: default None
        only write primary dihedra with ref db angle std dev greater than this
        value

    **Default values**:

    Data averaged from Sep 2019 Dunbrack cullpdb_pc20_res2.2_R1.0.

    Please see

    `PISCES: A Protein Sequence Culling Server <https://dunbrack.fccc.edu/pisces/>`_

    'G. Wang and R. L. Dunbrack, Jr. PISCES: a protein sequence culling
    server. Bioinformatics, 19:1589-1591, 2003.'

    'primary' and 'secondary' dihedra are defined in ic_data.py.  Specifically,
    secondary dihedra can be determined as a fixed rotation from another known
    angle, for example N-Ca-C-O can be estimated from N-Ca-C-N (psi).

    Standard deviations are listed in
    <biopython distribution>/Bio/PDB/ic_data.py for default values, and can be
    used to limit which hedra and dihedra are defaulted vs. output exact
    measurements from structure (see hCut and pCut above).  Default values for
    primary dihedra (psi, phi, omega, chi1, etc.) are chosen as the most common
    integer value, not an average.

    :raises PDBException: if entity level is A (Atom)
    :raises Exception: if entity does not have .level attribute
    """
    enumerate_atoms(entity)
    with as_handle(file, 'w') as fp:
        try:
            if 'A' == entity.level:
                raise PDBException('No PIC output at Atom level')
            elif 'R' == entity.level:
                if 2 == entity.is_disordered():
                    for r in entity.child_dict.values():
                        _wpr(r, fp, pdbid, chainid, picFlags=picFlags, hCut=hCut, pCut=pCut)
                else:
                    _wpr(entity, fp, pdbid, chainid, picFlags=picFlags, hCut=hCut, pCut=pCut)
            elif 'C' == entity.level:
                if not chainid:
                    chainid = entity.id
                for res in entity:
                    write_PIC(res, fp, pdbid, chainid, picFlags=picFlags, hCut=hCut, pCut=pCut)
            elif 'M' == entity.level:
                for chn in entity:
                    write_PIC(chn, fp, pdbid, chainid, picFlags=picFlags, hCut=hCut, pCut=pCut)
            elif 'S' == entity.level:
                if not pdbid:
                    pdbid = entity.header.get('idcode', None)
                hdr = entity.header.get('head', None)
                dd = pdb_date(entity.header.get('deposition_date', None))
                if hdr:
                    fp.write('HEADER    {:40}{:8}   {:4}\n'.format(hdr.upper(), dd or '', pdbid or ''))
                nam = entity.header.get('name', None)
                if nam:
                    fp.write('TITLE     ' + nam.upper() + '\n')
                for mdl in entity:
                    write_PIC(mdl, fp, pdbid, chainid, picFlags=picFlags, hCut=hCut, pCut=pCut)
            else:
                raise PDBException('Cannot identify level: ' + str(entity.level))
        except KeyError:
            raise Exception('write_PIC: argument is not a Biopython PDB Entity ' + str(entity))