import os
from copy import deepcopy
from nibabel import load, funcs, Nifti1Image
import numpy as np
from ..interfaces.base import (
from ..utils.filemanip import ensure_list, save_json, split_filename
from ..utils.misc import find_indices, normalize_mc_params
from .. import logging, config
class StimulusCorrelation(BaseInterface):
    """Determines if stimuli are correlated with motion or intensity
    parameters.

    Currently this class supports an SPM generated design matrix and requires
    intensity parameters. This implies that one must run
    :ref:`ArtifactDetect <nipype.algorithms.rapidart.ArtifactDetect>`
    and :ref:`Level1Design <nipype.interfaces.spm.model.Level1Design>` prior to
    running this or provide an SPM.mat file and intensity parameters through
    some other means.

    Examples
    --------

    >>> sc = StimulusCorrelation()
    >>> sc.inputs.realignment_parameters = 'functional.par'
    >>> sc.inputs.intensity_values = 'functional.rms'
    >>> sc.inputs.spm_mat_file = 'SPM.mat'
    >>> sc.inputs.concatenated_design = False
    >>> sc.run() # doctest: +SKIP

    """
    input_spec = StimCorrInputSpec
    output_spec = StimCorrOutputSpec

    def _get_output_filenames(self, motionfile, output_dir):
        """Generate output files based on motion filenames

        Parameters
        ----------
        motionfile: file/string
            Filename for motion parameter file
        output_dir: string
            output directory in which the files will be generated
        """
        _, filename = os.path.split(motionfile)
        filename, _ = os.path.splitext(filename)
        corrfile = os.path.join(output_dir, ''.join(('qa.', filename, '_stimcorr.txt')))
        return corrfile

    def _stimcorr_core(self, motionfile, intensityfile, designmatrix, cwd=None):
        """
        Core routine for determining stimulus correlation

        """
        if not cwd:
            cwd = os.getcwd()
        mc_in = np.loadtxt(motionfile)
        g_in = np.loadtxt(intensityfile)
        g_in.shape = (g_in.shape[0], 1)
        dcol = designmatrix.shape[1]
        mccol = mc_in.shape[1]
        concat_matrix = np.hstack((np.hstack((designmatrix, mc_in)), g_in))
        cm = np.corrcoef(concat_matrix, rowvar=0)
        corrfile = self._get_output_filenames(motionfile, cwd)
        file = open(corrfile, 'w')
        file.write('Stats for:\n')
        file.write('Stimulus correlated motion:\n%s\n' % motionfile)
        for i in range(dcol):
            file.write('SCM.%d:' % i)
            for v in cm[i, dcol + np.arange(mccol)]:
                file.write(' %.2f' % v)
            file.write('\n')
        file.write('Stimulus correlated intensity:\n%s\n' % intensityfile)
        for i in range(dcol):
            file.write('SCI.%d: %.2f\n' % (i, cm[i, -1]))
        file.close()

    def _get_spm_submatrix(self, spmmat, sessidx, rows=None):
        """
        Parameters
        ----------
        spmmat: scipy matlab object
            full SPM.mat file loaded into a scipy object
        sessidx: int
            index to session that needs to be extracted.
        """
        designmatrix = spmmat['SPM'][0][0].xX[0][0].X
        U = spmmat['SPM'][0][0].Sess[0][sessidx].U[0]
        if rows is None:
            rows = spmmat['SPM'][0][0].Sess[0][sessidx].row[0] - 1
        cols = spmmat['SPM'][0][0].Sess[0][sessidx].col[0][list(range(len(U)))] - 1
        outmatrix = designmatrix.take(rows.tolist(), axis=0).take(cols.tolist(), axis=1)
        return outmatrix

    def _run_interface(self, runtime):
        """Execute this module."""
        import scipy.io as sio
        motparamlist = self.inputs.realignment_parameters
        intensityfiles = self.inputs.intensity_values
        spmmat = sio.loadmat(self.inputs.spm_mat_file, struct_as_record=False)
        nrows = []
        for i in range(len(motparamlist)):
            sessidx = i
            rows = None
            if self.inputs.concatenated_design:
                sessidx = 0
                mc_in = np.loadtxt(motparamlist[i])
                rows = np.sum(nrows) + np.arange(mc_in.shape[0])
                nrows.append(mc_in.shape[0])
            matrix = self._get_spm_submatrix(spmmat, sessidx, rows)
            self._stimcorr_core(motparamlist[i], intensityfiles[i], matrix, os.getcwd())
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        files = []
        for i, f in enumerate(self.inputs.realignment_parameters):
            files.insert(i, self._get_output_filenames(f, os.getcwd()))
        if files:
            outputs['stimcorr_files'] = files
        return outputs