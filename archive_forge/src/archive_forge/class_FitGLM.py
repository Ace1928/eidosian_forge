import os
from .base import NipyBaseInterface
from ..base import (
class FitGLM(NipyBaseInterface):
    """
    Fit GLM model based on the specified design. Supports only single or concatenated runs.
    """
    input_spec = FitGLMInputSpec
    output_spec = FitGLMOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nb
        import numpy as np
        import nipy.modalities.fmri.glm as GLM
        import nipy.modalities.fmri.design_matrix as dm
        try:
            BlockParadigm = dm.BlockParadigm
        except AttributeError:
            from nipy.modalities.fmri.experimental_paradigm import BlockParadigm
        session_info = self.inputs.session_info
        functional_runs = self.inputs.session_info[0]['scans']
        if isinstance(functional_runs, (str, bytes)):
            functional_runs = [functional_runs]
        nii = nb.load(functional_runs[0])
        data = nii.get_fdata(caching='unchanged')
        if isdefined(self.inputs.mask):
            mask = np.asanyarray(nb.load(self.inputs.mask).dataobj) > 0
        else:
            mask = np.ones(nii.shape[:3]) == 1
        timeseries = data[mask, :]
        del data
        for functional_run in functional_runs[1:]:
            nii = nb.load(functional_run, mmap=False)
            npdata = np.asarray(nii.dataobj)
            timeseries = np.concatenate((timeseries, npdata[mask, :]), axis=1)
            del npdata
        nscans = timeseries.shape[1]
        if 'hpf' in list(session_info[0].keys()):
            hpf = session_info[0]['hpf']
            drift_model = self.inputs.drift_model
        else:
            hpf = 0
            drift_model = 'Blank'
        reg_names = []
        for reg in session_info[0]['regress']:
            reg_names.append(reg['name'])
        reg_vals = np.zeros((nscans, len(reg_names)))
        for i in range(len(reg_names)):
            reg_vals[:, i] = np.array(session_info[0]['regress'][i]['val']).reshape(1, -1)
        frametimes = np.linspace(0, (nscans - 1) * self.inputs.TR, nscans)
        conditions = []
        onsets = []
        duration = []
        for i, cond in enumerate(session_info[0]['cond']):
            onsets += cond['onset']
            conditions += [cond['name']] * len(cond['onset'])
            if len(cond['duration']) == 1:
                duration += cond['duration'] * len(cond['onset'])
            else:
                duration += cond['duration']
        if conditions:
            paradigm = BlockParadigm(con_id=conditions, onset=onsets, duration=duration)
        else:
            paradigm = None
        design_matrix, self._reg_names = dm.dmtx_light(frametimes, paradigm, drift_model=drift_model, hfcut=hpf, hrf_model=self.inputs.hrf_model, add_regs=reg_vals, add_reg_names=reg_names)
        if self.inputs.normalize_design_matrix:
            for i in range(len(self._reg_names) - 1):
                design_matrix[:, i] = (design_matrix[:, i] - design_matrix[:, i].mean()) / design_matrix[:, i].std()
        if self.inputs.plot_design_matrix:
            import pylab
            pylab.pcolor(design_matrix)
            pylab.savefig('design_matrix.pdf')
            pylab.close()
            pylab.clf()
        glm = GLM.GeneralLinearModel()
        glm.fit(timeseries.T, design_matrix, method=self.inputs.method, model=self.inputs.model)
        self._beta_file = os.path.abspath('beta.nii')
        beta = np.zeros(mask.shape + (glm.beta.shape[0],))
        beta[mask, :] = glm.beta.T
        nb.save(nb.Nifti1Image(beta, nii.affine), self._beta_file)
        self._s2_file = os.path.abspath('s2.nii')
        s2 = np.zeros(mask.shape)
        s2[mask] = glm.s2
        nb.save(nb.Nifti1Image(s2, nii.affine), self._s2_file)
        if self.inputs.save_residuals:
            explained = np.dot(design_matrix, glm.beta)
            residuals = np.zeros(mask.shape + (nscans,))
            residuals[mask, :] = timeseries - explained.T
            self._residuals_file = os.path.abspath('residuals.nii')
            nb.save(nb.Nifti1Image(residuals, nii.affine), self._residuals_file)
        self._nvbeta = glm.nvbeta
        self._dof = glm.dof
        self._constants = glm._constants
        self._axis = glm._axis
        if self.inputs.model == 'ar1':
            self._a_file = os.path.abspath('a.nii')
            a = np.zeros(mask.shape)
            a[mask] = glm.a.squeeze()
            nb.save(nb.Nifti1Image(a, nii.affine), self._a_file)
        self._model = glm.model
        self._method = glm.method
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['beta'] = self._beta_file
        outputs['nvbeta'] = self._nvbeta
        outputs['s2'] = self._s2_file
        outputs['dof'] = self._dof
        outputs['constants'] = self._constants
        outputs['axis'] = self._axis
        outputs['reg_names'] = self._reg_names
        if self.inputs.model == 'ar1':
            outputs['a'] = self._a_file
        if self.inputs.save_residuals:
            outputs['residuals'] = self._residuals_file
        return outputs