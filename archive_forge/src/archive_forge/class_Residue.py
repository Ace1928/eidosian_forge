from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.PDB.Entity import Entity, DisorderedEntityWrapper
class Residue(Entity):
    """Represents a residue. A Residue object stores atoms."""

    def __init__(self, id, resname, segid):
        """Initialize the class."""
        self.level = 'R'
        self.disordered = 0
        self.resname = resname
        self.segid = segid
        self.internal_coord = None
        Entity.__init__(self, id)

    def __repr__(self):
        """Return the residue full id."""
        resname = self.get_resname()
        hetflag, resseq, icode = self.get_id()
        full_id = (resname, hetflag, resseq, icode)
        return '<Residue %s het=%s resseq=%s icode=%s>' % full_id

    def add(self, atom):
        """Add an Atom object.

        Checks for adding duplicate atoms, and raises a
        PDBConstructionException if so.
        """
        atom_id = atom.get_id()
        if self.has_id(atom_id):
            raise PDBConstructionException(f'Atom {atom_id} defined twice in residue {self}')
        Entity.add(self, atom)

    def flag_disordered(self):
        """Set the disordered flag."""
        self.disordered = 1

    def is_disordered(self):
        """Return 1 if the residue contains disordered atoms."""
        return self.disordered

    def get_resname(self):
        """Return the residue name."""
        return self.resname

    def get_unpacked_list(self):
        """Return the list of all atoms, unpack DisorderedAtoms."""
        atom_list = self.get_list()
        undisordered_atom_list = []
        for atom in atom_list:
            if atom.is_disordered():
                undisordered_atom_list += atom.disordered_get_list()
            else:
                undisordered_atom_list.append(atom)
        return undisordered_atom_list

    def get_segid(self):
        """Return the segment identifier."""
        return self.segid

    def get_atoms(self):
        """Return atoms."""
        yield from self