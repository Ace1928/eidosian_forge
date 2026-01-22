from copy import deepcopy
import csv, math, os
from nibabel import load
import numpy as np
from ..interfaces.base import (
from ..utils.filemanip import ensure_list
from ..utils.misc import normalize_mc_params
from .. import config, logging
def _concatenate_info(self, infolist):
    nscans = []
    for i, f in enumerate(self.inputs.functional_runs):
        if isinstance(f, list):
            numscans = len(f)
        elif isinstance(f, (str, bytes)):
            img = load(f)
            numscans = img.shape[3]
        else:
            raise Exception('Functional input not specified correctly')
        nscans.insert(i, numscans)
    infoout = infolist[0]
    for j, val in enumerate(infolist[0].durations):
        if len(infolist[0].onsets[j]) > 1 and len(val) == 1:
            infoout.durations[j] = infolist[0].durations[j] * len(infolist[0].onsets[j])
    for i, info in enumerate(infolist[1:]):
        if info.onsets:
            for j, val in enumerate(info.onsets):
                if self.inputs.input_units == 'secs':
                    onsets = np.array(info.onsets[j]) + self.inputs.time_repetition * sum(nscans[0:i + 1])
                    infoout.onsets[j].extend(onsets.tolist())
                else:
                    onsets = np.array(info.onsets[j]) + sum(nscans[0:i + 1])
                    infoout.onsets[j].extend(onsets.tolist())
            for j, val in enumerate(info.durations):
                if len(info.onsets[j]) > 1 and len(val) == 1:
                    infoout.durations[j].extend(info.durations[j] * len(info.onsets[j]))
                elif len(info.onsets[j]) == len(val):
                    infoout.durations[j].extend(info.durations[j])
                else:
                    raise ValueError('Mismatch in number of onsets and                                           durations for run {0}, condition                                           {1}'.format(i + 2, j + 1))
            if hasattr(info, 'amplitudes') and info.amplitudes:
                for j, val in enumerate(info.amplitudes):
                    infoout.amplitudes[j].extend(info.amplitudes[j])
            if hasattr(info, 'pmod') and info.pmod:
                for j, val in enumerate(info.pmod):
                    if val:
                        for key, data in enumerate(val.param):
                            infoout.pmod[j].param[key].extend(data)
        if hasattr(info, 'regressors') and info.regressors:
            for j, v in enumerate(info.regressors):
                infoout.regressors[j].extend(info.regressors[j])
        if not hasattr(infoout, 'regressors') or not infoout.regressors:
            infoout.regressors = []
        onelist = np.zeros((1, sum(nscans)))
        onelist[0, sum(nscans[0:i]):sum(nscans[0:i + 1])] = 1
        infoout.regressors.insert(len(infoout.regressors), onelist.tolist()[0])
    return ([infoout], nscans)