from ase.calculators.calculator import CalculatorError
class KIMCalculatorError(CalculatorError):
    """
    Indicates an error occurred in initializing an applicable
    calculator.  This either results from incompatible combinations of
    argument values passed to kim.KIM(), or from models that are
    incompatible in some way with this calculator
    """
    pass