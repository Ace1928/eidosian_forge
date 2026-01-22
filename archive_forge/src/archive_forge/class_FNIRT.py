import os
import os.path as op
from warnings import warn
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import split_filename
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class FNIRT(FSLCommand):
    """FSL FNIRT wrapper for non-linear registration

    For complete details, see the `FNIRT Documentation.
    <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FNIRT>`_

    Examples
    --------
    >>> from nipype.interfaces import fsl
    >>> from nipype.testing import example_data
    >>> fnt = fsl.FNIRT(affine_file=example_data('trans.mat'))
    >>> res = fnt.run(ref_file=example_data('mni.nii', in_file=example_data('structural.nii')) #doctest: +SKIP

    T1 -> Mni153

    >>> from nipype.interfaces import fsl
    >>> fnirt_mprage = fsl.FNIRT()
    >>> fnirt_mprage.inputs.in_fwhm = [8, 4, 2, 2]
    >>> fnirt_mprage.inputs.subsampling_scheme = [4, 2, 1, 1]

    Specify the resolution of the warps

    >>> fnirt_mprage.inputs.warp_resolution = (6, 6, 6)
    >>> res = fnirt_mprage.run(in_file='structural.nii', ref_file='mni.nii', warped_file='warped.nii', fieldcoeff_file='fieldcoeff.nii')#doctest: +SKIP

    We can check the command line and confirm that it's what we expect.

    >>> fnirt_mprage.cmdline  #doctest: +SKIP
    'fnirt --cout=fieldcoeff.nii --in=structural.nii --infwhm=8,4,2,2 --ref=mni.nii --subsamp=4,2,1,1 --warpres=6,6,6 --iout=warped.nii'

    """
    _cmd = 'fnirt'
    input_spec = FNIRTInputSpec
    output_spec = FNIRTOutputSpec
    filemap = {'warped_file': 'warped', 'field_file': 'field', 'jacobian_file': 'field_jacobian', 'modulatedref_file': 'modulated', 'out_intensitymap_file': 'intmap', 'log_file': 'log.txt', 'fieldcoeff_file': 'fieldwarp'}

    def _list_outputs(self):
        outputs = self.output_spec().get()
        for key, suffix in list(self.filemap.items()):
            inval = getattr(self.inputs, key)
            change_ext = True
            if key in ['warped_file', 'log_file']:
                if suffix.endswith('.txt'):
                    change_ext = False
                if isdefined(inval):
                    outputs[key] = os.path.abspath(inval)
                else:
                    outputs[key] = self._gen_fname(self.inputs.in_file, suffix='_' + suffix, change_ext=change_ext)
            elif isdefined(inval):
                if isinstance(inval, bool):
                    if inval:
                        outputs[key] = self._gen_fname(self.inputs.in_file, suffix='_' + suffix, change_ext=change_ext)
                else:
                    outputs[key] = os.path.abspath(inval)
            if key == 'out_intensitymap_file' and isdefined(outputs[key]):
                basename = FNIRT.intensitymap_file_basename(outputs[key])
                outputs[key] = [outputs[key], '%s.txt' % basename]
        return outputs

    def _format_arg(self, name, spec, value):
        if name in ('in_intensitymap_file', 'out_intensitymap_file'):
            if name == 'out_intensitymap_file':
                value = self._list_outputs()[name]
            value = [FNIRT.intensitymap_file_basename(v) for v in value]
            assert len(set(value)) == 1, 'Found different basenames for {}: {}'.format(name, value)
            return spec.argstr % value[0]
        if name in list(self.filemap.keys()):
            return spec.argstr % self._list_outputs()[name]
        return super(FNIRT, self)._format_arg(name, spec, value)

    def _gen_filename(self, name):
        if name in ['warped_file', 'log_file']:
            return self._list_outputs()[name]
        return None

    def write_config(self, configfile):
        """Writes out currently set options to specified config file

        XX TODO : need to figure out how the config file is written

        Parameters
        ----------
        configfile : /path/to/configfile
        """
        try:
            fid = open(configfile, 'w+')
        except IOError:
            print('unable to create config_file %s' % configfile)
        for item in list(self.inputs.get().items()):
            fid.write('%s\n' % item)
        fid.close()

    @classmethod
    def intensitymap_file_basename(cls, f):
        """Removes valid intensitymap extensions from `f`, returning a basename
        that can refer to both intensitymap files.
        """
        for ext in list(Info.ftypes.values()) + ['.txt']:
            if f.endswith(ext):
                return f[:-len(ext)]
        return f