import numpy as np
from ase.calculators.calculator import Calculator
from ase.calculators.calculator import compare_atoms
from . import kimpy_wrappers
from . import neighborlist
class KIMModelData:
    """Initializes and subsequently stores the KIM API Portable Model
    object, KIM API ComputeArguments object, and the neighbor list
    object used by instances of KIMModelCalculator.  Also stores the
    arrays which are registered in the KIM API and which are used to
    communicate with the model.
    """

    def __init__(self, model_name, ase_neigh, neigh_skin_ratio, debug=False):
        self.model_name = model_name
        self.ase_neigh = ase_neigh
        self.debug = debug
        self.init_kim()
        model_influence_dist = self.kim_model.get_influence_distance()
        model_cutoffs, padding_not_require_neigh = self.kim_model.get_neighbor_list_cutoffs_and_hints()
        self.species_map = self.create_species_map()
        self.init_neigh(neigh_skin_ratio, model_influence_dist, model_cutoffs, padding_not_require_neigh)

    def init_kim(self):
        """Create the KIM API Portable Model object and KIM API ComputeArguments
        object
        """
        if self.kim_initialized:
            return
        self.kim_model = kimpy_wrappers.PortableModel(self.model_name, self.debug)
        self.compute_args = self.kim_model.compute_arguments_create()

    def init_neigh(self, neigh_skin_ratio, model_influence_dist, model_cutoffs, padding_not_require_neigh):
        """Initialize neighbor list, either an ASE-native neighborlist
        or one created using the neighlist module in kimpy
        """
        neigh_list_object_type = neighborlist.ASENeighborList if self.ase_neigh else neighborlist.KimpyNeighborList
        self.neigh = neigh_list_object_type(self.compute_args, neigh_skin_ratio, model_influence_dist, model_cutoffs, padding_not_require_neigh, self.debug)

    def update_compute_args_pointers(self, energy, forces):
        self.compute_args.update(self.num_particles, self.species_code, self.particle_contributing, self.coords, energy, forces)

    def create_species_map(self):
        """Get all the supported species of the KIM model and the
        corresponding integer codes used by the model

        Returns
        -------
        species_map : dict
            key : str
                chemical symbols (e.g. "Ar")
            value : int
                species integer code (e.g. 1)
        """
        supported_species, codes = self.get_model_supported_species_and_codes()
        species_map = dict()
        for i, spec in enumerate(supported_species):
            species_map[spec] = codes[i]
            if self.debug:
                print('Species {} is supported and its code is: {}'.format(spec, codes[i]))
        return species_map

    @property
    def padding_image_of(self):
        return self.neigh.padding_image_of

    @property
    def num_particles(self):
        return self.neigh.num_particles

    @property
    def coords(self):
        return self.neigh.coords

    @property
    def particle_contributing(self):
        return self.neigh.particle_contributing

    @property
    def species_code(self):
        return self.neigh.species_code

    @property
    def kim_initialized(self):
        return hasattr(self, 'kim_model')

    @property
    def neigh_initialized(self):
        return hasattr(self, 'neigh')

    @property
    def get_model_supported_species_and_codes(self):
        return self.kim_model.get_model_supported_species_and_codes