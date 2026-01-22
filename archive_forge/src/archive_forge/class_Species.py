from ase.calculators.calculator import Parameters
class Species(Parameters):
    """
    Parameters for specifying the behaviour for a single species in the
    calculation. If the tag argument is set to an integer then atoms with
    the specified element and tag will be a separate species.

    Pseudopotential and basis set can be specified. Additionally the species
    can be set be a ghost species, meaning that they will not be considered
    atoms, but the corresponding basis set will be used.
    """

    def __init__(self, symbol, basis_set='DZP', pseudopotential=None, tag=None, ghost=False, excess_charge=None):
        kwargs = locals()
        kwargs.pop('self')
        Parameters.__init__(self, **kwargs)