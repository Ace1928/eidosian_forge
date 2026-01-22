from __future__ import print_function, absolute_import, division
from collections import defaultdict
import numpy as np
from pyodesys.util import import_
from pyodesys.core import integrate_chained
from pyodesys.symbolic import ScaledSys, TransformedSys, symmetricsys, PartiallySolvedSystem, get_logexp
from pyodesys.tests.test_core import (
from pyodesys.tests.bateman import bateman_full  # analytic, never mind the details
from pyodesys.tests.test_symbolic import decay_rhs, decay_dydt_factory, _get_decay3
def _get_transformed_partially_solved_system(NativeSys, multiple=False):
    odesys = _get_decay3()
    if multiple:

        def substitutions(x0, y0, p0, be):
            analytic0 = y0[0] * be.exp(-p0[0] * (odesys.indep - x0))
            analytic2 = y0[0] + y0[1] + y0[2] - analytic0 - odesys.dep[1]
            return {odesys.dep[0]: analytic0, odesys.dep[2]: analytic2}
    else:

        def substitutions(x0, y0, p0, be):
            return {odesys.dep[0]: y0[0] * sp.exp(-p0[0] * (odesys.indep - x0))}
    partsys = PartiallySolvedSystem(odesys, substitutions)

    class TransformedNativeSys(TransformedSys, NativeSys):
        pass
    LogLogSys = symmetricsys(get_logexp(), get_logexp(), SuperClass=TransformedNativeSys)
    return LogLogSys.from_other(partsys)