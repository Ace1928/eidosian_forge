import numpy as np
from ase.calculators.calculator import Calculator
from ase.calculators.calculator import compare_atoms
from . import kimpy_wrappers
from . import neighborlist
class KIMModelCalculator(Calculator):
    """Calculator that works with KIM Portable Models (PMs).

    Calculator that carries out direct communication between ASE and a
    KIM Portable Model (PM) through the kimpy library (which provides a
    set of python bindings to the KIM API).

    Parameters
    ----------
    model_name : str
      The unique identifier assigned to the interatomic model (for
      details, see https://openkim.org/doc/schema/kim-ids)

    ase_neigh : bool, optional
      False (default): Use kimpy's neighbor list library

      True: Use ASE's internal neighbor list mechanism (usually slower
      than the kimpy neighlist library)

    neigh_skin_ratio : float, optional
      Used to determine the neighbor list cutoff distance, r_neigh,
      through the relation r_neigh = (1 + neigh_skin_ratio) * rcut,
      where rcut is the model's influence distance. (Default: 0.2)

    release_GIL : bool, optional
      Whether to release python GIL.  Releasing the GIL allows a KIM
      model to run with multiple concurrent threads. (Default: False)

    debug : bool, optional
      If True, detailed information is printed to stdout. (Default:
      False)
    """
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, model_name, ase_neigh=False, neigh_skin_ratio=0.2, release_GIL=False, debug=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.release_GIL = release_GIL
        self.debug = debug
        if neigh_skin_ratio < 0:
            raise ValueError('Argument "neigh_skin_ratio" must be non-negative')
        self.energy = None
        self.forces = None
        self.kimmodeldata = KIMModelData(self.model_name, ase_neigh, neigh_skin_ratio, self.debug)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, traceback):
        pass

    def __repr__(self):
        return 'KIMModelCalculator(model_name={})'.format(self.model_name)

    def calculate(self, atoms=None, properties=['energy', 'forces', 'stress'], system_changes=['positions', 'numbers', 'cell', 'pbc']):
        """
        Inherited method from the ase Calculator class that is called by
        get_property()

        Parameters
        ----------
        atoms : Atoms
            Atoms object whose properties are desired

        properties : list of str
            List of what needs to be calculated.  Can be any combination
            of 'energy', 'forces' and 'stress'.

        system_changes : list of str
            List of what has changed since last calculation.  Can be any
            combination of these six: 'positions', 'numbers', 'cell',
            and 'pbc'.
        """
        Calculator.calculate(self, atoms, properties, system_changes)
        if system_changes:
            if self.need_neigh_update(atoms, system_changes):
                self.update_neigh(atoms, self.species_map)
                self.energy = np.array([0.0], dtype=np.double)
                self.forces = np.zeros([self.num_particles[0], 3], dtype=np.double)
                self.update_compute_args_pointers(self.energy, self.forces)
            else:
                self.update_kim_coords(atoms)
            self.kim_model.compute(self.compute_args, self.release_GIL)
        energy = self.energy[0]
        forces = self.assemble_padding_forces()
        try:
            volume = atoms.get_volume()
            stress = self.compute_virial_stress(self.forces, self.coords, volume)
        except ValueError:
            stress = None
        self.results['energy'] = energy
        self.results['free_energy'] = energy
        self.results['forces'] = forces
        self.results['stress'] = stress

    def check_state(self, atoms, tol=1e-15):
        return compare_atoms(self.atoms, atoms, excluded_properties={'initial_charges', 'initial_magmoms'})

    def assemble_padding_forces(self):
        """
        Assemble forces on padding atoms back to contributing atoms.

        Parameters
        ----------
        forces : 2D array of doubles
            Forces on both contributing and padding atoms

        num_contrib:  int
            Number of contributing atoms

        padding_image_of : 1D array of int
            Atom number, of which the padding atom is an image


        Returns
        -------
            Total forces on contributing atoms.
        """
        total_forces = np.array(self.forces[:self.num_contributing_particles])
        if self.padding_image_of.size != 0:
            pad_forces = self.forces[self.num_contributing_particles:]
            for f, org_index in zip(pad_forces, self.padding_image_of):
                total_forces[org_index] += f
        return total_forces

    @staticmethod
    def compute_virial_stress(forces, coords, volume):
        """Compute the virial stress in Voigt notation.

        Parameters
        ----------
        forces : 2D array
            Partial forces on all atoms (padding included)

        coords : 2D array
            Coordinates of all atoms (padding included)

        volume : float
            Volume of cell

        Returns
        -------
        stress : 1D array
            stress in Voigt order (xx, yy, zz, yz, xz, xy)
        """
        stress = np.zeros(6)
        stress[0] = -np.dot(forces[:, 0], coords[:, 0]) / volume
        stress[1] = -np.dot(forces[:, 1], coords[:, 1]) / volume
        stress[2] = -np.dot(forces[:, 2], coords[:, 2]) / volume
        stress[3] = -np.dot(forces[:, 1], coords[:, 2]) / volume
        stress[4] = -np.dot(forces[:, 0], coords[:, 2]) / volume
        stress[5] = -np.dot(forces[:, 0], coords[:, 1]) / volume
        return stress

    def get_model_supported_species_and_codes(self):
        return self.kimmodeldata.get_model_supported_species_and_codes

    @property
    def update_compute_args_pointers(self):
        return self.kimmodeldata.update_compute_args_pointers

    @property
    def kim_model(self):
        return self.kimmodeldata.kim_model

    @property
    def compute_args(self):
        return self.kimmodeldata.compute_args

    @property
    def num_particles(self):
        return self.kimmodeldata.num_particles

    @property
    def coords(self):
        return self.kimmodeldata.coords

    @property
    def padding_image_of(self):
        return self.kimmodeldata.padding_image_of

    @property
    def species_map(self):
        return self.kimmodeldata.species_map

    @property
    def neigh(self):
        return self.kimmodeldata.neigh

    @property
    def num_contributing_particles(self):
        return self.neigh.num_contributing_particles

    @property
    def update_kim_coords(self):
        return self.neigh.update_kim_coords

    @property
    def need_neigh_update(self):
        return self.neigh.need_neigh_update

    @property
    def update_neigh(self):
        return self.neigh.update