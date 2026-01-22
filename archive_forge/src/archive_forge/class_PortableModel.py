import functools
import kimpy
from .exceptions import KIMModelNotFound, KIMModelInitializationError, KimpyError
class PortableModel:
    """Creates a KIM API Portable Model object and provides a minimal interface to it"""

    def __init__(self, model_name, debug):
        self.model_name = model_name
        self.debug = debug
        units_accepted, self.kim_model = model_create(kimpy.numbering.zeroBased, kimpy.length_unit.A, kimpy.energy_unit.eV, kimpy.charge_unit.e, kimpy.temperature_unit.K, kimpy.time_unit.ps, self.model_name)
        if not units_accepted:
            raise KIMModelInitializationError('Requested units not accepted in kimpy.model.create')
        if self.debug:
            l_unit, e_unit, c_unit, te_unit, ti_unit = check_call(self.kim_model.get_units)
            print('Length unit is: {}'.format(l_unit))
            print('Energy unit is: {}'.format(e_unit))
            print('Charge unit is: {}'.format(c_unit))
            print('Temperature unit is: {}'.format(te_unit))
            print('Time unit is: {}'.format(ti_unit))
            print()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, traceback):
        pass

    def get_model_supported_species_and_codes(self):
        """Get all of the supported species for this model and their
        corresponding integer codes that are defined in the KIM API

        Returns
        -------
        species : list of str
            Abbreviated chemical symbols of all species the mmodel
            supports (e.g. ["Mo", "S"])

        codes : list of int
            Integer codes used by the model for each species (order
            corresponds to the order of ``species``)
        """
        species = []
        codes = []
        num_kim_species = kimpy.species_name.get_number_of_species_names()
        for i in range(num_kim_species):
            species_name = get_species_name(i)
            species_support, code = self.get_species_support_and_code(species_name)
            if species_support:
                species.append(str(species_name))
                codes.append(code)
        return (species, codes)

    @check_call_wrapper
    def compute(self, compute_args_wrapped, release_GIL):
        return self.kim_model.compute(compute_args_wrapped.compute_args, release_GIL)

    @check_call_wrapper
    def get_species_support_and_code(self, species_name):
        return self.kim_model.get_species_support_and_code(species_name)

    @check_call_wrapper
    def get_influence_distance(self):
        return self.kim_model.get_influence_distance()

    @check_call_wrapper
    def get_neighbor_list_cutoffs_and_hints(self):
        return self.kim_model.get_neighbor_list_cutoffs_and_hints()

    def compute_arguments_create(self):
        return ComputeArguments(self, self.debug)

    @property
    def initialized(self):
        return hasattr(self, 'kim_model')