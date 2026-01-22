import warnings
from Bio.Data.PDBData import nucleic_letters_3to1
from Bio.Data.PDBData import nucleic_letters_3to1_extended
from Bio.Data.PDBData import protein_letters_3to1
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.PDB.PDBExceptions import PDBException
from Bio.PDB.vectors import calc_dihedral, calc_angle
from Bio.Seq import Seq
class _PPBuilder:
    """Base class to extract polypeptides.

    It checks if two consecutive residues in a chain are connected.
    The connectivity test is implemented by a subclass.

    This assumes you want both standard and non-standard amino acids.
    """

    def __init__(self, radius):
        """Initialize the base class.

        :param radius: distance
        :type radius: float
        """
        self.radius = radius

    def _accept(self, residue, standard_aa_only):
        """Check if the residue is an amino acid (PRIVATE)."""
        if is_aa(residue, standard=standard_aa_only):
            return True
        elif not standard_aa_only and 'CA' in residue.child_dict:
            warnings.warn('Assuming residue %s is an unknown modified amino acid' % residue.get_resname())
            return True
        else:
            return False

    def build_peptides(self, entity, aa_only=1):
        """Build and return a list of Polypeptide objects.

        :param entity: polypeptides are searched for in this object
        :type entity: L{Structure}, L{Model} or L{Chain}

        :param aa_only: if 1, the residue needs to be a standard AA
        :type aa_only: int
        """
        is_connected = self._is_connected
        accept = self._accept
        level = entity.get_level()
        if level == 'S':
            model = entity[0]
            chain_list = model.get_list()
        elif level == 'M':
            chain_list = entity.get_list()
        elif level == 'C':
            chain_list = [entity]
        else:
            raise PDBException('Entity should be Structure, Model or Chain.')
        pp_list = []
        for chain in chain_list:
            chain_it = iter(chain)
            try:
                prev_res = next(chain_it)
                while not accept(prev_res, aa_only):
                    prev_res = next(chain_it)
            except StopIteration:
                continue
            pp = None
            for next_res in chain_it:
                if accept(prev_res, aa_only) and accept(next_res, aa_only) and is_connected(prev_res, next_res):
                    if pp is None:
                        pp = Polypeptide()
                        pp.append(prev_res)
                        pp_list.append(pp)
                    pp.append(next_res)
                else:
                    pp = None
                prev_res = next_res
        return pp_list