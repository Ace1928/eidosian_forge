from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.PDB.Entity import Entity, DisorderedEntityWrapper
class DisorderedResidue(DisorderedEntityWrapper):
    """DisorderedResidue is a wrapper around two or more Residue objects.

    It is used to represent point mutations (e.g. there is a Ser 60 and a Cys 60
    residue, each with 50 % occupancy).
    """

    def __init__(self, id):
        """Initialize the class."""
        DisorderedEntityWrapper.__init__(self, id)

    def __repr__(self):
        """Return disordered residue full identifier."""
        if self.child_dict:
            resname = self.get_resname()
            hetflag, resseq, icode = self.get_id()
            full_id = (resname, hetflag, resseq, icode)
            return '<DisorderedResidue %s het=%s resseq=%i icode=%s>' % full_id
        else:
            return '<Empty DisorderedResidue>'

    def add(self, atom):
        """Add atom to residue."""
        residue = self.disordered_get()
        if atom.is_disordered() != 2:
            resname = residue.get_resname()
            het, resseq, icode = residue.get_id()
            residue.add(atom)
            raise PDBConstructionException('Blank altlocs in duplicate residue %s (%s, %i, %s)' % (resname, het, resseq, icode))
        residue.add(atom)

    def sort(self):
        """Sort the atoms in the child Residue objects."""
        for residue in self.disordered_get_list():
            residue.sort()

    def disordered_add(self, residue):
        """Add a residue object and use its resname as key.

        Arguments:
         - residue - Residue object

        """
        resname = residue.get_resname()
        chain = self.get_parent()
        residue.set_parent(chain)
        assert not self.disordered_has_id(resname)
        self[resname] = residue
        self.disordered_select(resname)

    def disordered_remove(self, resname):
        """Remove a child residue from the DisorderedResidue.

        Arguments:
         - resname - name of the child residue to remove, as a string.

        """
        residue = self.child_dict[resname]
        is_selected = self.selected_child is residue
        del self.child_dict[resname]
        residue.detach_parent()
        if is_selected and self.child_dict:
            child = next(iter(self.child_dict))
            self.disordered_select(child)
        elif not self.child_dict:
            self.selected_child = None