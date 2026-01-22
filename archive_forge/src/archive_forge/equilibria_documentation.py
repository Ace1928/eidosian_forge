import warnings
import numpy as np
from .chemistry import equilibrium_quotient, Equilibrium, Species
from .reactionsystem import ReactionSystem
from ._util import get_backend
from .util.pyutil import deprecated
from ._eqsys import EqCalcResult, NumSysLin, NumSysLog, NumSysSquare as _NumSysSquare

        Parameters
        ----------
        init_concs : array or dict
        varied_data : array
        varied_idx : int or str
        x0 : array
        NumSys : _NumSys subclass
            See :class:`NumSysLin`, :class:`NumSysLog`, etc.
        plot_kwargs : dict
            See py:meth:`pyneqsys.NeqSys.solve`. Two additional keys
            are intercepted here:
                latex_names: bool (default: False)
                conc_unit_str: str (default: 'M')
        neqsys_type : str
            what method to use for NeqSys construction (get_neqsys_*)
        \*\*kwargs :
            Keyword arguments passed on to py:meth:`pyneqsys.NeqSys.solve_series`.

        