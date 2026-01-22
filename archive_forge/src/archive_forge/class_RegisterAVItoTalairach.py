import os
import os.path
from ... import logging
from ...utils.filemanip import split_filename, copyfile
from .base import (
from ..base import isdefined, TraitedSpec, File, traits, Directory
class RegisterAVItoTalairach(FSScriptCommand):
    """
    converts the vox2vox from talairach_avi to a talairach.xfm file

    This is a script that converts the vox2vox from talairach_avi to a
    talairach.xfm file. It is meant to replace the following cmd line:

    tkregister2_cmdl         --mov $InVol         --targ $FREESURFER_HOME/average/mni305.cor.mgz         --xfmout ${XFM}         --vox2vox talsrcimg_to_${target}_t4_vox2vox.txt         --noedit         --reg talsrcimg.reg.tmp.dat
    set targ = $FREESURFER_HOME/average/mni305.cor.mgz
    set subject = mgh-02407836-v2
    set InVol = $SUBJECTS_DIR/$subject/mri/orig.mgz
    set vox2vox = $SUBJECTS_DIR/$subject/mri/transforms/talsrcimg_to_711-2C_as_mni_average_305_t4_vox2vox.txt

    Examples
    ========

    >>> from nipype.interfaces.freesurfer import RegisterAVItoTalairach
    >>> register = RegisterAVItoTalairach()
    >>> register.inputs.in_file = 'structural.mgz'                         # doctest: +SKIP
    >>> register.inputs.target = 'mni305.cor.mgz'                          # doctest: +SKIP
    >>> register.inputs.vox2vox = 'talsrcimg_to_structural_t4_vox2vox.txt' # doctest: +SKIP
    >>> register.cmdline                                                   # doctest: +SKIP
    'avi2talxfm structural.mgz mni305.cor.mgz talsrcimg_to_structural_t4_vox2vox.txt talairach.auto.xfm'

    >>> register.run() # doctest: +SKIP
    """
    _cmd = 'avi2talxfm'
    input_spec = RegisterAVItoTalairachInputSpec
    output_spec = RegisterAVItoTalairachOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs