from .backend import xrange
from .libmpf import (
from .libelefun import (
from .gammazeta import mpf_gamma, mpf_rgamma, mpf_loggamma, mpc_loggamma

    Convert a mpi interval to a string.

    **Arguments**

    *dps*
        decimal places to use for printing
    *use_spaces*
        use spaces for more readable output, defaults to true
    *brackets*
        pair of strings (or two-character string) giving left and right brackets
    *mode*
        mode of display: 'plusminus', 'percent', 'brackets' (default) or 'diff'
    *error_dps*
        limit the error to *error_dps* digits (mode 'plusminus and 'percent')

    Additional keyword arguments are forwarded to the mpf-to-string conversion
    for the components of the output.

    **Examples**

        >>> from mpmath import mpi, mp
        >>> mp.dps = 30
        >>> x = mpi(1, 2)._mpi_
        >>> mpi_to_str(x, 2, mode='plusminus')
        '1.5 +- 0.5'
        >>> mpi_to_str(x, 2, mode='percent')
        '1.5 (33.33%)'
        >>> mpi_to_str(x, 2, mode='brackets')
        '[1.0, 2.0]'
        >>> mpi_to_str(x, 2, mode='brackets' , brackets=('<', '>'))
        '<1.0, 2.0>'
        >>> x = mpi('5.2582327113062393041', '5.2582327113062749951')._mpi_
        >>> mpi_to_str(x, 15, mode='diff')
        '5.2582327113062[4, 7]'
        >>> mpi_to_str(mpi(0)._mpi_, 2, mode='percent')
        '0.0 (0.0%)'

    