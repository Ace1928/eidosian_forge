from ase.calculators.calculator import CalculatorError
class KIMModelNotFound(CalculatorError):
    """
    Requested model cannot be found in any of the KIM API model
    collections on the system
    """
    pass