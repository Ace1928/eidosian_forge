import os
import os.path as op
from warnings import warn
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import split_filename
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class FIRST(FSLCommand):
    """FSL run_first_all wrapper for segmentation of subcortical volumes

    http://www.fmrib.ox.ac.uk/fsl/first/index.html

    Examples
    --------

    >>> from nipype.interfaces import fsl
    >>> first = fsl.FIRST()
    >>> first.inputs.in_file = 'structural.nii'
    >>> first.inputs.out_file = 'segmented.nii'
    >>> res = first.run() #doctest: +SKIP

    """
    _cmd = 'run_first_all'
    input_spec = FIRSTInputSpec
    output_spec = FIRSTOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if isdefined(self.inputs.list_of_specific_structures):
            structures = self.inputs.list_of_specific_structures
        else:
            structures = ['L_Hipp', 'R_Hipp', 'L_Accu', 'R_Accu', 'L_Amyg', 'R_Amyg', 'L_Caud', 'R_Caud', 'L_Pall', 'R_Pall', 'L_Puta', 'R_Puta', 'L_Thal', 'R_Thal', 'BrStem']
        outputs['original_segmentations'] = self._gen_fname('original_segmentations')
        outputs['segmentation_file'] = self._gen_fname('segmentation_file')
        outputs['vtk_surfaces'] = self._gen_mesh_names('vtk_surfaces', structures)
        outputs['bvars'] = self._gen_mesh_names('bvars', structures)
        return outputs

    def _gen_fname(self, basename):
        path, outname, ext = split_filename(self.inputs.out_file)
        method = 'none'
        if isdefined(self.inputs.method) and self.inputs.method != 'none':
            method = 'fast'
            if self.inputs.list_of_specific_structures and self.inputs.method == 'auto':
                method = 'none'
        if isdefined(self.inputs.method_as_numerical_threshold):
            thres = '%.4f' % self.inputs.method_as_numerical_threshold
            method = thres.replace('.', '')
        if basename == 'original_segmentations':
            return op.abspath('%s_all_%s_origsegs.nii.gz' % (outname, method))
        if basename == 'segmentation_file':
            return op.abspath('%s_all_%s_firstseg.nii.gz' % (outname, method))
        return None

    def _gen_mesh_names(self, name, structures):
        path, prefix, ext = split_filename(self.inputs.out_file)
        if name == 'vtk_surfaces':
            vtks = list()
            for struct in structures:
                vtk = prefix + '-' + struct + '_first.vtk'
                vtks.append(op.abspath(vtk))
            return vtks
        if name == 'bvars':
            bvars = list()
            for struct in structures:
                bvar = prefix + '-' + struct + '_first.bvars'
                bvars.append(op.abspath(bvar))
            return bvars
        return None