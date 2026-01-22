import functools
import kimpy
from .exceptions import KIMModelNotFound, KIMModelInitializationError, KimpyError
class SimulatorModel:
    """Creates a KIM API Simulator Model object and provides a minimal
    interface to it.  This is only necessary in this package in order to
    extract any information about a given simulator model because it is
    generally embedded in a shared object.
    """

    def __init__(self, model_name):
        self.model_name = model_name
        self.simulator_model = simulator_model_create(self.model_name)
        self.simulator_model.close_template_map()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, traceback):
        pass

    @property
    def simulator_name(self):
        simulator_name, _ = self.simulator_model.get_simulator_name_and_version()
        return simulator_name

    @property
    def num_supported_species(self):
        num_supported_species = self.simulator_model.get_number_of_supported_species()
        if num_supported_species == 0:
            raise KIMModelInitializationError('Unable to determine supported species of simulator model {}.'.format(self.model_name))
        else:
            return num_supported_species

    @property
    def supported_species(self):
        supported_species = []
        for spec_code in range(self.num_supported_species):
            species = check_call(self.simulator_model.get_supported_species, spec_code)
            supported_species.append(species)
        return tuple(supported_species)

    @property
    def num_metadata_fields(self):
        return self.simulator_model.get_number_of_simulator_fields()

    @property
    def metadata(self):
        sm_metadata_fields = {}
        for field in range(self.num_metadata_fields):
            extent, field_name = check_call(self.simulator_model.get_simulator_field_metadata, field)
            sm_metadata_fields[field_name] = []
            for ln in range(extent):
                field_line = check_call(self.simulator_model.get_simulator_field_line, field, ln)
                sm_metadata_fields[field_name].append(field_line)
        return sm_metadata_fields

    @property
    def supported_units(self):
        try:
            supported_units = self.metadata['units'][0]
        except (KeyError, IndexError):
            raise KIMModelInitializationError('Unable to determine supported units of simulator model {}.'.format(self.model_name))
        return supported_units

    @property
    def atom_style(self):
        """
        See if a 'model-init' field exists in the SM metadata and, if
        so, whether it contains any entries including an "atom_style"
        command.  This is specific to LAMMPS SMs and is only required
        for using the LAMMPSrun calculator because it uses
        lammps.inputwriter to create a data file.  All other content in
        'model-init', if it exists, is ignored.
        """
        atom_style = None
        for ln in self.metadata.get('model-init', []):
            if ln.find('atom_style') != -1:
                atom_style = ln.split()[1]
        return atom_style

    @property
    def model_defn(self):
        return self.metadata['model-defn']

    @property
    def initialized(self):
        return hasattr(self, 'simulator_model')