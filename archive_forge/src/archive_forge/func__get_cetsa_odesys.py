from __future__ import (absolute_import, division, print_function)
import math
from collections import OrderedDict
from ..util import import_
import numpy as np
from ..symbolic import SymbolicSys, TransformedSys, symmetricsys
def _get_cetsa_odesys(molar_unitless, loglog, NativeSys=None, explicit_NL=False, MySys=None):
    names = 'N U NL L A'.split()
    params = OrderedDict([('T', 'T'), ('Ha_f', '\\Delta_f\\ H^\\neq'), ('Sa_f', '\\Delta_f S^\\neq'), ('dCp_u', '\\Delta_u\\ C_p'), ('He_u', '\\Delta_u H'), ('Tm_C', 'T_{m(C)}'), ('Ha_agg', '\\Delta_{agg}\\ H^\\neq'), ('Sa_agg', '\\Delta_{agg}\\ S^\\neq'), ('dCp_dis', '\\Delta_{dis}\\ C_p'), ('Ha_as', '\\Delta_{as}\\ H^\\neq'), ('Sa_as', '\\Delta_{as}\\ S^\\neq'), ('He_dis', '\\Delta_{dis}\\ H'), ('Se_dis', '\\Delta_{dis}\\ S')])
    param_keys = list(params.keys())

    def Eyring(dH, dS, T, R, kB_over_h, be):
        return kB_over_h * T * be.exp(-(dH - T * dS) / (R * T))

    def get_rates(x, y, p, be=math, T0=298.15, T0C=273.15, R=8.3144598, kB_over_h=1.38064852e-23 / 6.62607004e-34):
        pd = dict(zip(param_keys, p))
        He_u_T = pd['He_u'] + pd['dCp_u'] * (pd['T'] - T0)
        He_dis_T = pd['He_dis'] + pd['dCp_dis'] * (pd['T'] - T0)
        Se_u = pd['He_u'] / (T0C + pd['Tm_C']) + pd['dCp_u'] * be.log(pd['T'] / T0)
        Se_dis = pd['Se_dis'] + pd['dCp_dis'] * be.log(pd['T'] / T0)

        def C(k):
            return y[names.index(k)]
        return {'unfold': C('N') * Eyring(He_u_T + pd['Ha_f'], pd['Sa_f'] + Se_u, pd['T'], R, kB_over_h, be), 'fold': C('U') * Eyring(pd['Ha_f'], pd['Sa_f'], pd['T'], R, kB_over_h, be), 'aggregate': C('U') * Eyring(pd['Ha_agg'], pd['Sa_agg'], pd['T'], R, kB_over_h, be), 'dissociate': C('NL') * Eyring(He_dis_T + pd['Ha_as'], Se_dis + pd['Sa_as'], pd['T'], R, kB_over_h, be), 'associate': C('N') * C('L') * Eyring(pd['Ha_as'], pd['Sa_as'], pd['T'], R, kB_over_h, be) / molar_unitless}

    def f(x, y, p, be=math):
        r = get_rates(x, y, p, be)
        dydx = {'N': r['fold'] - r['unfold'] + r['dissociate'] - r['associate'], 'U': r['unfold'] - r['fold'] - r['aggregate'], 'A': r['aggregate'], 'L': r['dissociate'] - r['associate'], 'NL': r['associate'] - r['dissociate']}
        return [dydx[k] for k in (names if explicit_NL else names[:-1])]
    if loglog:
        logexp = (sp.log, sp.exp)
        if NativeSys:

            class SuperClass(TransformedSys, NativeSys):
                pass
        else:
            SuperClass = TransformedSys
        MySys = symmetricsys(logexp, logexp, SuperClass=SuperClass, exprs_process_cb=lambda exprs: [sp.powsimp(expr.expand(), force=True) for expr in exprs])
    else:
        MySys = NativeSys or SymbolicSys
    return MySys.from_callback(f, len(names) - (0 if explicit_NL else 1), len(param_keys), names=names if explicit_NL else names[:-1])