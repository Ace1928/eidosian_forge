from sympy.core.numbers import (I, pi)
from sympy.functions.elementary.complexes import (Abs, im, re)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import Matrix
from sympy.simplify.simplify import simplify
from sympy.physics.quantum import TensorProduct
A polarizing beam splitter Jones matrix at angle `theta`.

    Parameters
    ==========

    J : SymPy Matrix
        A Jones matrix.
    Tp : numeric type or SymPy Symbol
        The transmissivity of the P-polarized component.
    Rs : numeric type or SymPy Symbol
        The reflectivity of the S-polarized component.
    Ts : numeric type or SymPy Symbol
        The transmissivity of the S-polarized component.
    Rp : numeric type or SymPy Symbol
        The reflectivity of the P-polarized component.
    phia : numeric type or SymPy Symbol
        The phase difference between transmitted and reflected component for
        output mode a.
    phib : numeric type or SymPy Symbol
        The phase difference between transmitted and reflected component for
        output mode b.


    Returns
    =======

    SymPy Matrix
        A 4x4 matrix representing the PBS. This matrix acts on a 4x1 vector
        whose first two entries are the Jones vector on one of the PBS ports,
        and the last two entries the Jones vector on the other port.

    Examples
    ========

    Generic polarizing beam-splitter.

    >>> from sympy import pprint, symbols
    >>> from sympy.physics.optics.polarization import polarizing_beam_splitter
    >>> Ts, Rs, Tp, Rp = symbols(r"Ts, Rs, Tp, Rp", positive=True)
    >>> phia, phib = symbols("phi_a, phi_b", real=True)
    >>> PBS = polarizing_beam_splitter(Tp, Rs, Ts, Rp, phia, phib)
    >>> pprint(PBS, use_unicode=False)
    [   ____                           ____                    ]
    [ \/ Tp            0           I*\/ Rp           0         ]
    [                                                          ]
    [                  ____                       ____  I*phi_a]
    [   0            \/ Ts            0      -I*\/ Rs *e       ]
    [                                                          ]
    [    ____                         ____                     ]
    [I*\/ Rp           0            \/ Tp            0         ]
    [                                                          ]
    [               ____  I*phi_b                    ____      ]
    [   0      -I*\/ Rs *e            0            \/ Ts       ]

    