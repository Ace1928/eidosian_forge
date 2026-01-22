from collections import defaultdict
import numpy as np
import kimpy
from kimpy import neighlist
from ase.neighborlist import neighbor_list
from ase import Atom
from .kimpy_wrappers import check_call_wrapper
class KimpyNeighborList(NeighborList):

    def __init__(self, compute_args, neigh_skin_ratio, model_influence_dist, model_cutoffs, padding_not_require_neigh, debug):
        super().__init__(neigh_skin_ratio, model_influence_dist, model_cutoffs, padding_not_require_neigh, debug)
        self.neigh = neighlist.NeighList()
        compute_args.set_callback_pointer(kimpy.compute_callback_name.GetNeighborList, neighlist.get_neigh_kim(), self.neigh)

    @check_call_wrapper
    def build(self):
        return self.neigh.build(self.coords, self.influence_dist, self.cutoffs, self.need_neigh)

    @check_call_wrapper
    def create_paddings(self, cell, pbc, contributing_coords, contributing_species_code):
        cell = np.asarray(cell, dtype=np.double)
        pbc = np.asarray(pbc, dtype=np.intc)
        contributing_coords = np.asarray(contributing_coords, dtype=np.double)
        return neighlist.create_paddings(self.influence_dist, cell, pbc, contributing_coords, contributing_species_code)

    def update(self, atoms, species_map):
        """Create the neighbor list along with the other required
        parameters (which are stored as instance attributes). The
        required parameters are:

            - num_particles
            - coords
            - particle_contributing
            - species_code

        Note that the KIM API requires a neighbor list that has indices
        corresponding to each atom.
        """
        cell = np.asarray(atoms.get_cell(), dtype=np.double)
        pbc = np.asarray(atoms.get_pbc(), dtype=np.intc)
        contributing_coords = np.asarray(atoms.get_positions(), dtype=np.double)
        self.num_contributing_particles = atoms.get_global_number_of_atoms()
        num_contributing = self.num_contributing_particles
        try:
            contributing_species_code = np.array([species_map[s] for s in atoms.get_chemical_symbols()], dtype=np.intc)
        except KeyError as e:
            raise RuntimeError('Species not supported by KIM model; {}'.format(str(e)))
        if pbc.any():
            padding_coords, padding_species_code, self.padding_image_of = self.create_paddings(cell, pbc, contributing_coords, contributing_species_code)
            num_padding = padding_species_code.size
            self.num_particles = [num_contributing + num_padding]
            self.coords = np.concatenate((contributing_coords, padding_coords))
            self.species_code = np.concatenate((contributing_species_code, padding_species_code))
            self.particle_contributing = [1] * num_contributing + [0] * num_padding
            self.need_neigh = [1] * self.num_particles[0]
            if not self.padding_need_neigh:
                self.need_neigh[num_contributing:] = 0
        else:
            self.padding_image_of = []
            self.num_particles = [num_contributing]
            self.coords = contributing_coords
            self.species_code = contributing_species_code
            self.particle_contributing = [1] * num_contributing
            self.need_neigh = self.particle_contributing
        self.build()
        self.last_update_positions = atoms.get_positions()
        if self.debug:
            print('Debug: called update_kimpy_neigh')
            print()