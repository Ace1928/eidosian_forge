import copy
import sys
import warnings
import numpy as np
from Bio.PDB.Entity import DisorderedEntityWrapper
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.vectors import Vector
from Bio.Data import IUPACData
class DisorderedAtom(DisorderedEntityWrapper):
    """Contains all Atom objects that represent the same disordered atom.

    One of these atoms is "selected" and all method calls not caught
    by DisorderedAtom are forwarded to the selected Atom object. In that way, a
    DisorderedAtom behaves exactly like a normal Atom. By default, the selected
    Atom object represents the Atom object with the highest occupancy, but a
    different Atom object can be selected by using the disordered_select(altloc)
    method.
    """

    def __init__(self, id):
        """Create DisorderedAtom.

        Arguments:
         - id - string, atom name

        """
        self.last_occupancy = -sys.maxsize
        DisorderedEntityWrapper.__init__(self, id)

    def __iter__(self):
        """Iterate through disordered atoms."""
        yield from self.disordered_get_list()

    def __repr__(self):
        """Return disordered atom identifier."""
        if self.child_dict:
            return f'<DisorderedAtom {self.get_id()}>'
        else:
            return f'<Empty DisorderedAtom {self.get_id()}>'

    def center_of_mass(self):
        """Return the center of mass of the DisorderedAtom as a numpy array.

        Assumes all child atoms have the same mass (same element).
        """
        children = self.disordered_get_list()
        if not children:
            raise ValueError(f'{self} does not have children')
        coords = np.asarray([a.coord for a in children], dtype=np.float32)
        return np.average(coords, axis=0, weights=None)

    def disordered_get_list(self):
        """Return list of atom instances.

        Sorts children by altloc (empty, then alphabetical).
        """
        return sorted(self.child_dict.values(), key=lambda a: ord(a.altloc))

    def disordered_add(self, atom):
        """Add a disordered atom."""
        atom.flag_disorder()
        residue = self.get_parent()
        atom.set_parent(residue)
        altloc = atom.get_altloc()
        occupancy = atom.get_occupancy()
        self[altloc] = atom
        if occupancy > self.last_occupancy:
            self.last_occupancy = occupancy
            self.disordered_select(altloc)

    def disordered_remove(self, altloc):
        """Remove a child atom altloc from the DisorderedAtom.

        Arguments:
         - altloc - name of the altloc to remove, as a string.

        """
        atom = self.child_dict[altloc]
        is_selected = self.selected_child is atom
        del self.child_dict[altloc]
        atom.detach_parent()
        if is_selected and self.child_dict:
            child = sorted(self.child_dict.values(), key=lambda a: a.occupancy)[-1]
            self.disordered_select(child.altloc)
        elif not self.child_dict:
            self.selected_child = None
            self.last_occupancy = -sys.maxsize

    def transform(self, rot, tran):
        """Apply rotation and translation to all children.

        See the documentation of Atom.transform for details.
        """
        for child in self:
            child.coord = np.dot(child.coord, rot) + tran