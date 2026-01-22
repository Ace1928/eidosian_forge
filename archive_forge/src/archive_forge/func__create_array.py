import numpy
from modin.error_message import ErrorMessage
from .arr import array
def _create_array(dtype, shape, order, subok, numpy_method):
    if order not in ['K', 'C']:
        ErrorMessage.single_warning("Array order besides 'C' is not currently supported in Modin. Defaulting to 'C' order.")
    if not subok:
        ErrorMessage.single_warning('Subclassing types is not currently supported in Modin. Defaulting to the same base dtype.')
    ErrorMessage.single_warning(f'np.{numpy_method}_like defaulting to NumPy.')
    return array(getattr(numpy, numpy_method)(shape, dtype=dtype))