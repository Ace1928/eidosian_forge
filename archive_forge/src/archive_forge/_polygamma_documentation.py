import cupy
from cupyx.scipy.special import _digamma
from cupyx.scipy.special import _gamma
from cupyx.scipy.special import _zeta
Polygamma function n.

    Args:
        n (cupy.ndarray): The order of the derivative of `psi`.
        x (cupy.ndarray): Where to evaluate the polygamma function.

    Returns:
        cupy.ndarray: The result.

    .. seealso:: :data:`scipy.special.polygamma`

    