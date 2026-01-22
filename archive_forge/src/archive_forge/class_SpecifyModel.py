from copy import deepcopy
import csv, math, os
from nibabel import load
import numpy as np
from ..interfaces.base import (
from ..utils.filemanip import ensure_list
from ..utils.misc import normalize_mc_params
from .. import config, logging
class SpecifyModel(BaseInterface):
    """
    Makes a model specification compatible with spm/fsl designers.

    The subject_info field should contain paradigm information in the form of
    a Bunch or a list of Bunch. The Bunch should contain the following
    information::

        [Mandatory]
        conditions : list of names
        onsets : lists of onsets corresponding to each condition
        durations : lists of durations corresponding to each condition. Should be
            left to a single 0 if all events are being modelled as impulses.

        [Optional]
        regressor_names : list of str
            list of names corresponding to each column. Should be None if
            automatically assigned.
        regressors : list of lists
            values for each regressor - must correspond to the number of
            volumes in the functional run
        amplitudes : lists of amplitudes for each event. This will be ignored by
            SPM's Level1Design.

        The following two (tmod, pmod) will be ignored by any Level1Design class
        other than SPM:

        tmod : lists of conditions that should be temporally modulated. Should
            default to None if not being used.
        pmod : list of Bunch corresponding to conditions
          - name : name of parametric modulator
          - param : values of the modulator
          - poly : degree of modulation

    Alternatively, you can provide information through event files.

    The event files have to be in 1, 2 or 3 column format with the columns
    corresponding to Onsets, Durations and Amplitudes and they have to have the
    name event_name.runXXX... e.g.: Words.run001.txt. The event_name part will
    be used to create the condition names.

    Examples
    --------
    >>> from nipype.algorithms import modelgen
    >>> from nipype.interfaces.base import Bunch
    >>> s = modelgen.SpecifyModel()
    >>> s.inputs.input_units = 'secs'
    >>> s.inputs.functional_runs = ['functional2.nii', 'functional3.nii']
    >>> s.inputs.time_repetition = 6
    >>> s.inputs.high_pass_filter_cutoff = 128.
    >>> evs_run2 = Bunch(conditions=['cond1'], onsets=[[2, 50, 100, 180]], durations=[[1]])
    >>> evs_run3 = Bunch(conditions=['cond1'], onsets=[[30, 40, 100, 150]], durations=[[1]])
    >>> s.inputs.subject_info = [evs_run2, evs_run3]

    >>> # Using pmod
    >>> evs_run2 = Bunch(conditions=['cond1', 'cond2'], onsets=[[2, 50], [100, 180]], durations=[[0], [0]], pmod=[Bunch(name=['amp'], poly=[2], param=[[1, 2]]), None])
    >>> evs_run3 = Bunch(conditions=['cond1', 'cond2'], onsets=[[20, 120], [80, 160]], durations=[[0], [0]], pmod=[Bunch(name=['amp'], poly=[2], param=[[1, 2]]), None])
    >>> s.inputs.subject_info = [evs_run2, evs_run3]

    """
    input_spec = SpecifyModelInputSpec
    output_spec = SpecifyModelOutputSpec

    def _generate_standard_design(self, infolist, functional_runs=None, realignment_parameters=None, outliers=None):
        """Generate a standard design matrix paradigm given information about each run."""
        sessinfo = []
        output_units = 'secs'
        if 'output_units' in self.inputs.traits():
            output_units = self.inputs.output_units
        for i, info in enumerate(infolist):
            sessinfo.insert(i, dict(cond=[]))
            if isdefined(self.inputs.high_pass_filter_cutoff):
                sessinfo[i]['hpf'] = float(self.inputs.high_pass_filter_cutoff)
            if hasattr(info, 'conditions') and info.conditions is not None:
                for cid, cond in enumerate(info.conditions):
                    sessinfo[i]['cond'].insert(cid, dict())
                    sessinfo[i]['cond'][cid]['name'] = info.conditions[cid]
                    scaled_onset = scale_timings(info.onsets[cid], self.inputs.input_units, output_units, self.inputs.time_repetition)
                    sessinfo[i]['cond'][cid]['onset'] = scaled_onset
                    scaled_duration = scale_timings(info.durations[cid], self.inputs.input_units, output_units, self.inputs.time_repetition)
                    sessinfo[i]['cond'][cid]['duration'] = scaled_duration
                    if hasattr(info, 'amplitudes') and info.amplitudes:
                        sessinfo[i]['cond'][cid]['amplitudes'] = info.amplitudes[cid]
                    if hasattr(info, 'tmod') and info.tmod and (len(info.tmod) > cid):
                        sessinfo[i]['cond'][cid]['tmod'] = info.tmod[cid]
                    if hasattr(info, 'pmod') and info.pmod and (len(info.pmod) > cid):
                        if info.pmod[cid]:
                            sessinfo[i]['cond'][cid]['pmod'] = []
                            for j, name in enumerate(info.pmod[cid].name):
                                sessinfo[i]['cond'][cid]['pmod'].insert(j, {})
                                sessinfo[i]['cond'][cid]['pmod'][j]['name'] = name
                                sessinfo[i]['cond'][cid]['pmod'][j]['poly'] = info.pmod[cid].poly[j]
                                sessinfo[i]['cond'][cid]['pmod'][j]['param'] = info.pmod[cid].param[j]
            sessinfo[i]['regress'] = []
            if hasattr(info, 'regressors') and info.regressors is not None:
                for j, r in enumerate(info.regressors):
                    sessinfo[i]['regress'].insert(j, dict(name='', val=[]))
                    if hasattr(info, 'regressor_names') and info.regressor_names is not None:
                        sessinfo[i]['regress'][j]['name'] = info.regressor_names[j]
                    else:
                        sessinfo[i]['regress'][j]['name'] = 'UR%d' % (j + 1)
                    sessinfo[i]['regress'][j]['val'] = info.regressors[j]
            sessinfo[i]['scans'] = functional_runs[i]
        if realignment_parameters is not None:
            for i, rp in enumerate(realignment_parameters):
                mc = realignment_parameters[i]
                for col in range(mc.shape[1]):
                    colidx = len(sessinfo[i]['regress'])
                    sessinfo[i]['regress'].insert(colidx, dict(name='', val=[]))
                    sessinfo[i]['regress'][colidx]['name'] = 'Realign%d' % (col + 1)
                    sessinfo[i]['regress'][colidx]['val'] = mc[:, col].tolist()
        if outliers is not None:
            for i, out in enumerate(outliers):
                numscans = 0
                for f in ensure_list(sessinfo[i]['scans']):
                    shape = load(f).shape
                    if len(shape) == 3 or shape[3] == 1:
                        iflogger.warning('You are using 3D instead of 4D files. Are you sure this was intended?')
                        numscans += 1
                    else:
                        numscans += shape[3]
                for j, scanno in enumerate(out):
                    colidx = len(sessinfo[i]['regress'])
                    sessinfo[i]['regress'].insert(colidx, dict(name='', val=[]))
                    sessinfo[i]['regress'][colidx]['name'] = 'Outlier%d' % (j + 1)
                    sessinfo[i]['regress'][colidx]['val'] = np.zeros((1, numscans))[0].tolist()
                    sessinfo[i]['regress'][colidx]['val'][int(scanno)] = 1
        return sessinfo

    def _generate_design(self, infolist=None):
        """Generate design specification for a typical fmri paradigm"""
        realignment_parameters = []
        if isdefined(self.inputs.realignment_parameters):
            for parfile in self.inputs.realignment_parameters:
                realignment_parameters.append(np.apply_along_axis(func1d=normalize_mc_params, axis=1, arr=np.loadtxt(parfile), source=self.inputs.parameter_source))
        outliers = []
        if isdefined(self.inputs.outlier_files):
            for filename in self.inputs.outlier_files:
                try:
                    outindices = np.loadtxt(filename, dtype=int)
                except IOError:
                    outliers.append([])
                else:
                    if outindices.size == 1:
                        outliers.append([outindices.tolist()])
                    else:
                        outliers.append(outindices.tolist())
        if infolist is None:
            if isdefined(self.inputs.subject_info):
                infolist = self.inputs.subject_info
            elif isdefined(self.inputs.event_files):
                infolist = gen_info(self.inputs.event_files)
            elif isdefined(self.inputs.bids_event_file):
                infolist = bids_gen_info(self.inputs.bids_event_file, self.inputs.bids_condition_column, self.inputs.bids_amplitude_column, self.inputs.time_repetition)
        self._sessinfo = self._generate_standard_design(infolist, functional_runs=self.inputs.functional_runs, realignment_parameters=realignment_parameters, outliers=outliers)

    def _run_interface(self, runtime):
        """ """
        self._sessioninfo = None
        self._generate_design()
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        if not hasattr(self, '_sessinfo'):
            self._generate_design()
        outputs['session_info'] = self._sessinfo
        return outputs