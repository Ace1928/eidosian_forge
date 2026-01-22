import warnings
import autoray as ar
import numpy as _np
from autograd.numpy.numpy_boxes import ArrayBox
from autoray import numpy as np
from . import single_dispatch  # pylint:disable=unused-import
def _get_interface_of_single_tensor(tensor):
    """Returns the name of the package that any array/tensor manipulations
    will dispatch to. The returned strings correspond to those used for PennyLane
    :doc:`interfaces </introduction/interfaces>`.

    Args:
        tensor (tensor_like): tensor input

    Returns:
        str: name of the interface

    **Example**

    >>> x = torch.tensor([1., 2.])
    >>> get_interface(x)
    'torch'
    >>> from pennylane import numpy as np
    >>> x = np.array([4, 5], requires_grad=True)
    >>> get_interface(x)
    'autograd'
    """
    namespace = tensor.__class__.__module__.split('.')[0]
    if namespace in ('pennylane', 'autograd'):
        return 'autograd'
    res = ar.infer_backend(tensor)
    if res == 'builtins':
        return 'numpy'
    return res