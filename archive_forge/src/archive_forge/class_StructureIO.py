import warnings
from Bio import BiopythonWarning
from Bio.PDB.PDBExceptions import PDBIOException
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.Data.IUPACData import atom_weights
class StructureIO:
    """Base class to derive structure file format writers from."""

    def __init__(self):
        """Initialise."""

    def set_structure(self, pdb_object):
        """Check what the user is providing and build a structure."""
        if pdb_object.level == 'S':
            structure = pdb_object
        else:
            sb = StructureBuilder()
            sb.init_structure('pdb')
            sb.init_seg(' ')
            if pdb_object.level == 'M':
                sb.structure.add(pdb_object.copy())
                self.structure = sb.structure
            else:
                sb.init_model(0)
                if pdb_object.level == 'C':
                    sb.structure[0].add(pdb_object.copy())
                else:
                    chain_id = 'A'
                    sb.init_chain(chain_id)
                    if pdb_object.level == 'R':
                        if pdb_object.parent is not None:
                            og_chain_id = pdb_object.parent.id
                            sb.structure[0][chain_id].id = og_chain_id
                            chain_id = og_chain_id
                        sb.structure[0][chain_id].add(pdb_object.copy())
                    else:
                        sb.init_residue('DUM', ' ', 1, ' ')
                        sb.structure[0][chain_id].child_list[0].add(pdb_object.copy())
                        try:
                            og_chain_id = pdb_object.parent.parent.id
                        except AttributeError:
                            pass
                        else:
                            sb.structure[0][chain_id].id = og_chain_id
            structure = sb.structure
        self.structure = structure