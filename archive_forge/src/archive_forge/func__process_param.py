import sys
import re
import copy
import logging
from pyomo.common.log import is_debug_set
from pyomo.common.collections import Bunch, OrderedDict
from pyomo.common.errors import ApplicationError
from pyomo.dataportal.parse_datacmds import parse_data_commands, _re_number
from pyomo.dataportal.factory import DataManagerFactory, UnknownDataManager
from pyomo.core.base.set import UnknownSetDimen
from pyomo.core.base.util import flatten_tuple
def _process_param(cmd, _model, _data, _default, index=None, param=None, ncolumns=None):
    """
    Called by _process_data to process data for a Parameter declaration
    """
    generate_debug_messages = is_debug_set(logger)
    if generate_debug_messages:
        logger.debug('DEBUG: _process_param(start) %s', cmd)
    dflt = None
    singledef = True
    cmd = cmd[1:]
    if cmd[0] == ':':
        singledef = False
        cmd = cmd[1:]
    if singledef:
        pname = cmd[0]
        cmd = cmd[1:]
        if len(cmd) >= 2 and cmd[0] == 'default':
            dflt = cmd[1]
            cmd = cmd[2:]
        if dflt != None:
            _default[pname] = dflt
        if cmd[0] == ':=':
            cmd = cmd[1:]
        transpose = False
        if cmd[0] == '(tr)':
            transpose = True
            if cmd[1] == ':':
                cmd = cmd[1:]
            else:
                cmd[0] = ':'
        if cmd[0] != ':':
            if generate_debug_messages:
                logger.debug('DEBUG: _process_param (singledef without :...:=) %s', cmd)
            cmd = _apply_templates(cmd)
            if not transpose:
                if pname not in _data:
                    _data[pname] = {}
                if not ncolumns is None:
                    finaldata = _process_data_list(pname, ncolumns - 1, cmd)
                elif not _model is None:
                    _param = getattr(_model, pname)
                    _dim = _param.dim()
                    if _dim is UnknownSetDimen:
                        _dim = _guess_set_dimen(_param.index_set())
                    finaldata = _process_data_list(pname, _dim, cmd)
                else:
                    _dim = 1 if len(cmd) > 1 else 0
                    finaldata = _process_data_list(pname, _dim, cmd)
                for key in finaldata:
                    _data[pname][key] = finaldata[key]
            else:
                tmp = ['param', pname, ':=']
                i = 1
                while i < len(cmd):
                    i0 = i
                    while cmd[i] != ':=':
                        i = i + 1
                    ncol = i - i0 + 1
                    lcmd = i
                    while lcmd < len(cmd) and cmd[lcmd] != ':':
                        lcmd += 1
                    j0 = i0 - 1
                    for j in range(1, ncol):
                        ii = 1 + i
                        kk = ii + j
                        while kk < lcmd:
                            if cmd[kk] != '.':
                                tmp.append(copy.copy(cmd[j + j0]))
                                tmp.append(copy.copy(cmd[ii]))
                                tmp.append(copy.copy(cmd[kk]))
                            ii = ii + ncol
                            kk = kk + ncol
                    i = lcmd + 1
                _process_param(tmp, _model, _data, _default, index=index, param=param, ncolumns=ncolumns)
        else:
            tmp = ['param', pname, ':=']
            if param is None:
                param = [pname]
            i = 1
            if generate_debug_messages:
                logger.debug('DEBUG: _process_param (singledef with :...:=) %s', cmd)
            while i < len(cmd):
                i0 = i
                while i < len(cmd) and cmd[i] != ':=':
                    i = i + 1
                if i == len(cmd):
                    raise ValueError('ERROR: Trouble on line ' + str(Lineno) + ' of file ' + Filename)
                ncol = i - i0 + 1
                lcmd = i
                while lcmd < len(cmd) and cmd[lcmd] != ':':
                    lcmd += 1
                j0 = i0 - 1
                for j in range(1, ncol):
                    ii = 1 + i
                    kk = ii + j
                    while kk < lcmd:
                        if cmd[kk] != '.':
                            if transpose:
                                tmp.append(copy.copy(cmd[j + j0]))
                                tmp.append(copy.copy(cmd[ii]))
                            else:
                                tmp.append(copy.copy(cmd[ii]))
                                tmp.append(copy.copy(cmd[j + j0]))
                            tmp.append(copy.copy(cmd[kk]))
                        ii = ii + ncol
                        kk = kk + ncol
                i = lcmd + 1
                _process_param(tmp, _model, _data, _default, index=index, param=param[0], ncolumns=3)
    else:
        if generate_debug_messages:
            logger.debug("DEBUG: _process_param (cmd[0]=='param:') %s", cmd)
        i = 0
        nsets = 0
        while i < len(cmd):
            if cmd[i] == ':=':
                i = -1
                break
            if cmd[i] == ':':
                nsets = i
                break
            i += 1
        nparams = 0
        _i = i + 1
        while i < len(cmd):
            if cmd[i] == ':=':
                nparams = i - _i
                break
            i += 1
        if i == len(cmd):
            raise ValueError('Trouble on data file line ' + str(Lineno) + ' of file ' + Filename)
        if generate_debug_messages:
            logger.debug('NSets %d', nsets)
        Lcmd = len(cmd)
        j = 0
        d = 1
        while j < nsets:
            sname = cmd[j]
            if not ncolumns is None:
                d = ncolumns - nparams
            elif _model is None:
                d = 1
            else:
                index = getattr(_model, sname)
                d = _guess_set_dimen(index)
            np = i - 1
            if generate_debug_messages:
                logger.debug('I %d, J %d, SName %s, d %d', i, j, sname, d)
            dnp = d + np - 1
            ii = i + j + 1
            tmp = ['set', cmd[j], ':=']
            while ii < Lcmd:
                if d > 1:
                    _tmp = []
                    for dd in range(0, d):
                        _tmp.append(copy.copy(cmd[ii + dd]))
                    tmp.append(tuple(_tmp))
                else:
                    for dd in range(0, d):
                        tmp.append(copy.copy(cmd[ii + dd]))
                ii += dnp
            _process_set(tmp, _model, _data)
            j += 1
        if nsets > 0:
            j += 1
        jstart = j
        if param is None:
            param = []
            _j = j
            while _j < i:
                param.append(cmd[_j])
                _j += 1
        while j < i:
            pname = param[j - jstart]
            if generate_debug_messages:
                logger.debug('I %d, J %d, Pname %s', i, j, pname)
            if not ncolumns is None:
                d = ncolumns - nparams
            elif _model is None:
                d = 1
            else:
                _param = getattr(_model, pname)
                d = _param.dim()
                if d is UnknownSetDimen:
                    d = _guess_set_dimen(_param.index_set())
            if nsets > 0:
                np = i - 1
                dnp = d + np - 1
                ii = i + 1
                kk = i + d + j - 1
            else:
                np = i
                dnp = d + np
                ii = i + 1
                kk = np + 1 + d + nsets + j
            tmp = ['param', pname, ':=']
            if generate_debug_messages:
                logger.debug('dnp %d\nnp %d', dnp, np)
            while kk < Lcmd:
                if generate_debug_messages:
                    logger.debug('kk %d, ii %d', kk, ii)
                iid = ii + d
                while ii < iid:
                    tmp.append(copy.copy(cmd[ii]))
                    ii += 1
                ii += dnp - d
                tmp.append(copy.copy(cmd[kk]))
                kk += dnp
            if not ncolumns is None:
                nc = ncolumns - nparams + 1
            else:
                nc = None
            _process_param(tmp, _model, _data, _default, index=index, param=param[j - jstart], ncolumns=nc)
            j += 1