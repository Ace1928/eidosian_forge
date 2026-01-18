from math import isclose
from sympy.core.numbers import I
from sympy.core.symbol import Dummy
from sympy.functions.elementary.complexes import (Abs, arg)
from sympy.functions.elementary.exponential import log
from sympy.abc import s, p, a
from sympy.external import import_module
from sympy.physics.control.control_plots import \
from sympy.physics.control.lti import (TransferFunction,
from sympy.testing.pytest import raises, skip
def test_pole_zero():
    if not numpy:
        skip('NumPy is required for this test')

    def pz_tester(sys, expected_value):
        z, p = pole_zero_numerical_data(sys)
        z_check = numpy.allclose(z, expected_value[0])
        p_check = numpy.allclose(p, expected_value[1])
        return p_check and z_check
    exp1 = [[], [-0.24999999999999994 + 1.3919410907075054j, -0.24999999999999994 - 1.3919410907075054j]]
    exp2 = [[0.0], [-0.25 + 0.3227486121839514j, -0.25 - 0.3227486121839514j]]
    exp3 = [[0.0], [-0.5000000000000004 + 0.8660254037844395j, -0.5000000000000004 - 0.8660254037844395j, 0.9999999999999998 + 0j]]
    exp4 = [[], [5.0, 0.0, 0.0, 0.0]]
    exp5 = [[-5.645751311064592, -0.5000000000000008, -0.3542486889354093], [-0.24999999999999986 + 1.3919410907075052j, -0.24999999999999986 - 1.3919410907075052j, -0.2499999999999998 + 0.32274861218395134j, -0.2499999999999998 - 0.32274861218395134j]]
    exp6 = [[], [-1.1641600331447917 - 3.545808351896439j, -0.8358399668552097 + 2.5458083518964383j]]
    assert pz_tester(tf1, exp1)
    assert pz_tester(tf2, exp2)
    assert pz_tester(tf3, exp3)
    assert pz_tester(ser1, exp4)
    assert pz_tester(par1, exp5)
    assert pz_tester(tf8, exp6)