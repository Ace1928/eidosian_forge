import warnings
from Bio import BiopythonWarning
from Bio.PDB.PDBExceptions import PDBIOException
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.Data.IUPACData import atom_weights
def _get_atom_line(self, atom, hetfield, segid, atom_number, resname, resseq, icode, chain_id, charge='  '):
    """Return an ATOM PDB string (PRIVATE)."""
    if hetfield != ' ':
        record_type = 'HETATM'
    else:
        record_type = 'ATOM  '
    try:
        atom_number = int(atom_number)
    except ValueError:
        raise ValueError(f'{atom_number!r} is not a number.Atom serial numbers must be numerical If you are converting from an mmCIF structure, try using preserve_atom_numbering=False')
    if atom_number > 99999:
        raise ValueError(f"Atom serial number ('{atom_number}') exceeds PDB format limit.")
    if atom.element:
        element = atom.element.strip().upper()
        if element.capitalize() not in atom_weights and element != 'X':
            raise ValueError(f'Unrecognised element {atom.element}')
        element = element.rjust(2)
    else:
        element = '  '
    name = atom.fullname.strip()
    if len(name) < 4 and name[:1].isalpha() and (len(element.strip()) < 2):
        name = ' ' + name
    altloc = atom.altloc
    x, y, z = atom.coord
    if not self.is_pqr:
        bfactor = atom.bfactor
        try:
            occupancy = f'{atom.occupancy:6.2f}'
        except (TypeError, ValueError):
            if atom.occupancy is None:
                occupancy = ' ' * 6
                warnings.warn(f'Missing occupancy in atom {atom.full_id!r} written as blank', BiopythonWarning)
            else:
                raise ValueError(f'Invalid occupancy value: {atom.occupancy!r}') from None
        args = (record_type, atom_number, name, altloc, resname, chain_id, resseq, icode, x, y, z, occupancy, bfactor, segid, element, charge)
        return _ATOM_FORMAT_STRING % args
    else:
        try:
            pqr_charge = f'{atom.pqr_charge:7.4f}'
        except (TypeError, ValueError):
            if atom.pqr_charge is None:
                pqr_charge = ' ' * 7
                warnings.warn(f'Missing PQR charge in atom {atom.full_id} written as blank', BiopythonWarning)
            else:
                raise ValueError(f'Invalid PQR charge value: {atom.pqr_charge!r}') from None
        try:
            radius = f'{atom.radius:6.4f}'
        except (TypeError, ValueError):
            if atom.radius is None:
                radius = ' ' * 6
                warnings.warn(f'Missing radius in atom {atom.full_id} written as blank', BiopythonWarning)
            else:
                raise ValueError(f'Invalid radius value: {atom.radius}') from None
        args = (record_type, atom_number, name, altloc, resname, chain_id, resseq, icode, x, y, z, pqr_charge, radius, element)
        return _PQR_ATOM_FORMAT_STRING % args