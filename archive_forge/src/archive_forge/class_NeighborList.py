from collections import defaultdict
import numpy as np
import kimpy
from kimpy import neighlist
from ase.neighborlist import neighbor_list
from ase import Atom
from .kimpy_wrappers import check_call_wrapper
class NeighborList:
    kimpy_arrays = {'num_particles': np.intc, 'coords': np.double, 'particle_contributing': np.intc, 'species_code': np.intc, 'cutoffs': np.double, 'padding_image_of': np.intc, 'need_neigh': np.intc}

    def __setattr__(self, name, value):
        """
        Override assignment to any of the attributes listed in
        kimpy_arrays to automatically cast the object to a numpy array.
        This is done to avoid a ton of explicit numpy.array() calls (and
        the possibility that we forget to do the cast).  It is important
        to use np.asarray() here instead of np.array() because using the
        latter will mean that incrementation (+=) will create a new
        object that the reference is bound to, which becomes a problem
        if update_compute_args isn't called to reregister the
        corresponding address with the KIM API.
        """
        if name in self.kimpy_arrays and value is not None:
            value = np.asarray(value, dtype=self.kimpy_arrays[name])
        self.__dict__[name] = value

    def __init__(self, neigh_skin_ratio, model_influence_dist, model_cutoffs, padding_not_require_neigh, debug):
        self.skin = neigh_skin_ratio * model_influence_dist
        self.influence_dist = model_influence_dist + self.skin
        self.cutoffs = model_cutoffs + self.skin
        self.padding_need_neigh = not padding_not_require_neigh.all()
        self.debug = debug
        if self.debug:
            print()
            print('Calculator skin: {}'.format(self.skin))
            print(f'Model influence distance: {model_influence_dist}')
            print('Calculator influence distance (including skin distance): {}'.format(self.influence_dist))
            print('Number of cutoffs: {}'.format(model_cutoffs.size))
            print('Model cutoffs: {}'.format(model_cutoffs))
            print('Calculator cutoffs (including skin distance): {}'.format(self.cutoffs))
            print('Model needs neighbors of padding atoms: {}'.format(self.padding_need_neigh))
            print()
        self.neigh = None
        self.num_contributing_particles = None
        self.padding_image_of = None
        self.num_particles = None
        self.coords = None
        self.particle_contributing = None
        self.species_code = None
        self.need_neigh = None
        self.last_update_positions = None

    def update_kim_coords(self, atoms):
        """Update atomic positions in self.coords, which is where the KIM
        API will look to find them in order to pass them to the model.
        """
        if self.padding_image_of.size != 0:
            disp_contrib = atoms.positions - self.coords[:len(atoms)]
            disp_pad = disp_contrib[self.padding_image_of]
            self.coords += np.concatenate((disp_contrib, disp_pad))
        else:
            np.copyto(self.coords, atoms.positions)
        if self.debug:
            print('Debug: called update_kim_coords')
            print()

    def need_neigh_update(self, atoms, system_changes):
        need_neigh_update = True
        if len(system_changes) == 1 and 'positions' in system_changes:
            if self.last_update_positions is not None:
                a = self.last_update_positions
                b = atoms.positions
                if a.shape == b.shape:
                    delta = np.linalg.norm(a - b, axis=1)
                    ind = np.argpartition(delta, -2)[-2:]
                    if sum(delta[ind]) <= self.skin:
                        need_neigh_update = False
        return need_neigh_update