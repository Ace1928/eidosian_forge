from Bio.File import as_handle
from Bio.PDB.PDBExceptions import PDBException
from Bio.PDB.internal_coords import IC_Residue, IC_Chain
from Bio.PDB.vectors import homog_scale_mtx
import numpy as np  # type: ignore
def write_SCAD(entity, file, scale=None, pdbid=None, backboneOnly=False, includeCode=True, maxPeptideBond=None, start=None, fin=None, handle='protein'):
    """Write hedron assembly to file as OpenSCAD matrices.

    This routine calls both :meth:`.IC_Chain.internal_to_atom_coordinates` and
    :meth:`.IC_Chain.atom_to_internal_coordinates` due to requirements for
    scaling, explicit bonds around rings, and setting the coordinate space of
    the output model.

    Output data format is primarily:

    - matrix for each hedron:
        len1, angle2, len3, atom covalent bond class, flags to indicate
        atom/bond represented in previous hedron (OpenSCAD very slow with
        redundant overlapping elements), flags for bond features
    - transform matrices to assemble each hedron into residue dihedra sets
    - transform matrices for each residue to position in chain

    OpenSCAD software is included in this Python file to process these
    matrices into a model suitable for a 3D printing project.

    :param entity: Biopython PDB :class:`.Structure` entity
        structure data to export
    :param file: Bipoython :func:`.as_handle` filename or open file pointer
        file to write data to
    :param float scale:
        units (usually mm) per angstrom for STL output, written in output
    :param str pdbid:
        PDB idcode, written in output. Defaults to '0PDB' if not supplied
        and no 'idcode' set in entity
    :param bool backboneOnly: default False.
        Do not output side chain data past Cbeta if True
    :param bool includeCode: default True.
        Include OpenSCAD software (inline below) so output file can be loaded
        into OpenSCAD; if False, output data matrices only
    :param float maxPeptideBond: Optional default None.
        Override the cut-off in IC_Chain class (default 1.4) for detecting
        chain breaks.  If your target has chain breaks, pass a large number
        here to create a very long 'bond' spanning the break.
    :param int start,fin: default None
        Parameters for internal_to_atom_coords() to limit chain segment.
    :param str handle: default 'protein'
        name for top level of generated OpenSCAD matrix structure

    See :meth:`.IC_Residue.set_flexible` to set flags for specific residues to
    have rotatable bonds, and :meth:`.IC_Residue.set_hbond` to include cavities
    for small magnets to work as hydrogen bonds.
    See <https://www.thingiverse.com/thing:3957471> for implementation example.

    The OpenSCAD code explicitly creates spheres and cylinders to
    represent atoms and bonds in a 3D model.  Options are available
    to support rotatable bonds and magnetic hydrogen bonds.

    Matrices are written to link, enumerate and describe residues,
    dihedra, hedra, and chains, mirroring contents of the relevant IC_*
    data structures.

    The OpenSCAD matrix of hedra has additional information as follows:

    * the atom and bond state (single, double, resonance) are logged
        so that covalent radii may be used for atom spheres in the 3D models

    * bonds and atoms are tracked so that each is only created once

    * bond options for rotation and magnet holders for hydrogen bonds
        may be specified (see :meth:`.IC_Residue.set_flexible` and
        :meth:`.IC_Residue.set_hbond` )

    Note the application of :data:`Bio.PDB.internal_coords.IC_Chain.MaxPeptideBond`
    :  missing residues may be linked (joining chain segments with arbitrarily
    long bonds) by setting this to a large value.

    Note this uses the serial assembly per residue, placing each residue at
    the origin and supplying the coordinate space transform to OpenaSCAD

    All ALTLOC (disordered) residues and atoms are written to the output
    model.  (see :data:`Bio.PDB.internal_coords.IC_Residue.no_altloc`)
    """
    if maxPeptideBond is not None:
        mpbStash = IC_Chain.MaxPeptideBond
        IC_Chain.MaxPeptideBond = float(maxPeptideBond)
    added_IC_Atoms = False
    if 'S' == entity.level or 'M' == entity.level:
        for chn in entity.get_chains():
            if not chn.internal_coord:
                chn.internal_coord = IC_Chain(chn)
                added_IC_Atoms = True
    elif 'C' == entity.level:
        if not entity.internal_coord:
            entity.internal_coord = IC_Chain(entity)
            added_IC_Atoms = True
    else:
        raise PDBException('level not S, M or C: ' + str(entity.level))
    if added_IC_Atoms:
        entity.atom_to_internal_coordinates()
    else:
        entity.internal_to_atom_coordinates(None)
    if scale is not None:
        scaleMtx = homog_scale_mtx(scale)
        if 'C' == entity.level:
            entity.internal_coord.atomArray = np.dot(entity.internal_coord.atomArray[:], scaleMtx)
            entity.internal_coord.hAtoms_needs_update[:] = True
            entity.internal_coord.scale = scale
        else:
            for chn in entity.get_chains():
                if hasattr(chn.internal_coord, 'atomArray'):
                    chn.internal_coord.atomArray = np.dot(chn.internal_coord.atomArray[:], scaleMtx)
                    chn.internal_coord.hAtoms_needs_update[:] = True
                    chn.internal_coord.scale = scale
    allBondsStash = IC_Residue._AllBonds
    IC_Residue._AllBonds = True
    if 'C' == entity.level:
        entity.internal_coord.ordered_aa_ic_list[0].hedra = {}
        delattr(entity.internal_coord, 'hAtoms_needs_update')
        delattr(entity.internal_coord, 'hedraLen')
    else:
        for chn in entity.get_chains():
            chn.internal_coord.ordered_aa_ic_list[0].hedra = {}
            delattr(chn.internal_coord, 'hAtoms_needs_update')
            delattr(chn.internal_coord, 'hedraLen')
    entity.atom_to_internal_coordinates()
    IC_Residue._AllBonds = allBondsStash
    entity.internal_to_atom_coordinates()
    with as_handle(file, 'w') as fp:
        if includeCode:
            fp.write(peptide_scad)
        if not pdbid and hasattr(entity, 'header'):
            pdbid = entity.header.get('idcode', None)
        if pdbid is None or '' == pdbid:
            pdbid = '0PDB'
        fp.write('protein = [ "' + pdbid + '", ' + str(scale) + ',  // ID, protein_scale\n')
        if 'S' == entity.level or 'M' == entity.level:
            for chn in entity.get_chains():
                fp.write(' [\n')
                chn.internal_coord._write_SCAD(fp, backboneOnly=backboneOnly, start=start, fin=fin)
                fp.write(' ]\n')
        elif 'C' == entity.level:
            fp.write(' [\n')
            entity.internal_coord._write_SCAD(fp, backboneOnly=backboneOnly, start=start, fin=fin)
            fp.write(' ]\n')
        elif 'R' == entity.level:
            raise NotImplementedError('writescad single residue not yet implemented.')
        fp.write('\n];\n')
    if maxPeptideBond is not None:
        IC_Chain.MaxPeptideBond = mpbStash