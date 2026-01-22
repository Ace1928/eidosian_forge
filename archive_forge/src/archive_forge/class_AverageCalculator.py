from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.calculator import PropertyNotImplementedError
class AverageCalculator(LinearCombinationCalculator):
    """AverageCalculator for equal summation of multiple calculators (for thermodynamic purposes)..
    """

    def __init__(self, calcs, atoms=None):
        """Implementation of average of calculators.

        calcs: list
            List of an arbitrary number of :mod:`ase.calculators` objects.
        atoms: Atoms object
            Optional :class:`~ase.Atoms` object to which the calculator will be attached.
        """
        n = len(calcs)
        if n == 0:
            raise ValueError('The value of the calcs must be a list of Calculators')
        weights = [1 / n] * n
        super().__init__(calcs, weights, atoms)