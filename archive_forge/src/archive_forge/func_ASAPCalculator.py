import re
import os
from ase.data import atomic_masses, atomic_numbers
from ase.calculators.lammpsrun import LAMMPS
from ase.calculators.lammpslib import LAMMPSlib
from ase.calculators.lammps import convert
from .kimmodel import KIMModelCalculator
from .exceptions import KIMCalculatorError
def ASAPCalculator(model_name, model_type, options, **kwargs):
    """
    Can be used with either Portable Models or Simulator Models
    """
    import asap3
    options_not_allowed = {'pm': ['name', 'verbose'], 'sm': ['Params']}
    _check_conflict_options(options, options_not_allowed[model_type], simulator='asap')
    if model_type == 'pm':
        return asap3.OpenKIMcalculator(name=model_name, verbose=kwargs['verbose'], **options)
    elif model_type == 'sm':
        model_defn = kwargs['model_defn']
        supported_units = kwargs['supported_units']
        if supported_units != 'ase':
            raise KIMCalculatorError('KIM Simulator Model units are "{}", but expected to be "ase" for ASAP.'.format(supported_units))
        if len(model_defn) == 0:
            raise KIMCalculatorError('model-defn is an empty list in metadata file of Simulator Model {}'.format(model_name))
        elif len(model_defn) > 1:
            raise KIMCalculatorError('model-defn should contain only one entry for an ASAP model (found {} lines)'.format(len(model_defn)))
        if '' in model_defn:
            raise KIMCalculatorError('model-defn contains an empty string in metadata file of Simulator Model {}'.format(model_name))
        model_defn = model_defn[0].strip()
        model_defn_is_valid = False
        if model_defn.startswith('EMT'):
            mobj = re.search('\\(([A-Za-z0-9_\\(\\)]+)\\)', model_defn)
            if mobj is None:
                asap_calc = asap3.EMT()
            else:
                pp = mobj.group(1)
                if pp.startswith('EMTRasmussenParameters'):
                    asap_calc = asap3.EMT(parameters=asap3.EMTRasmussenParameters())
                    model_defn_is_valid = True
                elif pp.startswith('EMTMetalGlassParameters'):
                    asap_calc = asap3.EMT(parameters=asap3.EMTMetalGlassParameters())
                    model_defn_is_valid = True
        if not model_defn_is_valid:
            raise KIMCalculatorError('Unknown model "{}" requested for simulator asap.'.format(model_defn))
        asap_calc.set_subtractE0(False)
        return asap_calc