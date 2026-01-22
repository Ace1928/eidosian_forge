import numpy as np
from scipy.odr._odrpack import Model
class _QuadraticModel(Model):
    """
    Quadratic model

    This model is defined by :math:`y = \\beta_0 x^2 + \\beta_1 x + \\beta_2`

    Examples
    --------
    We can calculate orthogonal distance regression with a quadratic model:

    >>> from scipy import odr
    >>> import numpy as np
    >>> x = np.linspace(0.0, 5.0)
    >>> y = 1.0 * x ** 2 + 2.0 * x + 3.0
    >>> data = odr.Data(x, y)
    >>> odr_obj = odr.ODR(data, odr.quadratic)
    >>> output = odr_obj.run()
    >>> print(output.beta)
    [1. 2. 3.]

    """

    def __init__(self):
        super().__init__(_quadratic, fjacd=_quad_fjd, fjacb=_quad_fjb, estimate=_quad_est, meta={'name': 'Quadratic', 'equ': 'y = B_0*x**2 + B_1*x + B_2', 'TeXequ': '$y = \\beta_0 x^2 + \\beta_1 x + \\beta_2'})