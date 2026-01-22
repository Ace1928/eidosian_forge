import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class SampleToSurface(FSCommand):
    """Sample a volume to the cortical surface using Freesurfer's mri_vol2surf.

    You must supply a sampling method, range, and units.  You can project
    either a given distance (in mm) or a given fraction of the cortical
    thickness at that vertex along the surface normal from the target surface,
    and then set the value of that vertex to be either the value at that point
    or the average or maximum value found along the projection vector.

    By default, the surface will be saved as a vector with a length equal to the
    number of vertices on the target surface.  This is not a problem for Freesurfer
    programs, but if you intend to use the file with interfaces to another package,
    you must set the ``reshape`` input to True, which will factor the surface vector
    into a matrix with dimensions compatible with proper Nifti files.

    Examples
    --------

    >>> import nipype.interfaces.freesurfer as fs
    >>> sampler = fs.SampleToSurface(hemi="lh")
    >>> sampler.inputs.source_file = "cope1.nii.gz"
    >>> sampler.inputs.reg_file = "register.dat"
    >>> sampler.inputs.sampling_method = "average"
    >>> sampler.inputs.sampling_range = 1
    >>> sampler.inputs.sampling_units = "frac"
    >>> sampler.cmdline  # doctest: +ELLIPSIS
    'mri_vol2surf --hemi lh --o ...lh.cope1.mgz --reg register.dat --projfrac-avg 1.000 --mov cope1.nii.gz'
    >>> res = sampler.run() # doctest: +SKIP

    """
    _cmd = 'mri_vol2surf'
    input_spec = SampleToSurfaceInputSpec
    output_spec = SampleToSurfaceOutputSpec

    def _format_arg(self, name, spec, value):
        if name == 'sampling_method':
            range = self.inputs.sampling_range
            units = self.inputs.sampling_units
            if units == 'mm':
                units = 'dist'
            if isinstance(range, tuple):
                range = '%.3f %.3f %.3f' % range
            else:
                range = '%.3f' % range
            method = dict(point='', max='-max', average='-avg')[value]
            return '--proj%s%s %s' % (units, method, range)
        if name == 'reg_header':
            return spec.argstr % self.inputs.subject_id
        if name == 'override_reg_subj':
            return spec.argstr % self.inputs.subject_id
        if name in ['hits_file', 'vox_file']:
            return spec.argstr % self._get_outfilename(name)
        if name == 'out_type':
            if isdefined(self.inputs.out_file):
                _, base, ext = split_filename(self._get_outfilename())
                if ext != filemap[value]:
                    if ext in filemap.values():
                        raise ValueError('Cannot create {} file with extension {}'.format(value, ext))
                    else:
                        logger.warning('Creating %s file with extension %s: %s%s', value, ext, base, ext)
            if value in implicit_filetypes:
                return ''
        if name == 'surf_reg':
            if value is True:
                return spec.argstr % 'sphere.reg'
        return super(SampleToSurface, self)._format_arg(name, spec, value)

    def _get_outfilename(self, opt='out_file'):
        outfile = getattr(self.inputs, opt)
        if not isdefined(outfile) or isinstance(outfile, bool):
            if isdefined(self.inputs.out_type):
                if opt == 'hits_file':
                    suffix = '_hits.' + filemap[self.inputs.out_type]
                else:
                    suffix = '.' + filemap[self.inputs.out_type]
            elif opt == 'hits_file':
                suffix = '_hits.mgz'
            else:
                suffix = '.mgz'
            outfile = fname_presuffix(self.inputs.source_file, newpath=os.getcwd(), prefix=self.inputs.hemi + '.', suffix=suffix, use_ext=False)
        return outfile

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = os.path.abspath(self._get_outfilename())
        hitsfile = self.inputs.hits_file
        if isdefined(hitsfile):
            outputs['hits_file'] = hitsfile
            if isinstance(hitsfile, bool):
                hitsfile = self._get_outfilename('hits_file')
        voxfile = self.inputs.vox_file
        if isdefined(voxfile):
            if isinstance(voxfile, bool):
                voxfile = fname_presuffix(self.inputs.source_file, newpath=os.getcwd(), prefix=self.inputs.hemi + '.', suffix='_vox.txt', use_ext=False)
            outputs['vox_file'] = voxfile
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]
        return None