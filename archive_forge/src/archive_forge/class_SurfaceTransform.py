import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class SurfaceTransform(FSCommand):
    """Transform a surface file from one subject to another via a spherical registration.

    Both the source and target subject must reside in your Subjects Directory,
    and they must have been processed with recon-all, unless you are transforming
    to one of the icosahedron meshes.

    Examples
    --------

    >>> from nipype.interfaces.freesurfer import SurfaceTransform
    >>> sxfm = SurfaceTransform()
    >>> sxfm.inputs.source_file = "lh.cope1.nii.gz"
    >>> sxfm.inputs.source_subject = "my_subject"
    >>> sxfm.inputs.target_subject = "fsaverage"
    >>> sxfm.inputs.hemi = "lh"
    >>> sxfm.run() # doctest: +SKIP

    """
    _cmd = 'mri_surf2surf'
    input_spec = SurfaceTransformInputSpec
    output_spec = SurfaceTransformOutputSpec

    def _format_arg(self, name, spec, value):
        if name == 'target_type':
            if isdefined(self.inputs.out_file):
                _, base, ext = split_filename(self._list_outputs()['out_file'])
                if ext != filemap[value]:
                    if ext in filemap.values():
                        raise ValueError('Cannot create {} file with extension {}'.format(value, ext))
                    else:
                        logger.warning('Creating %s file with extension %s: %s%s', value, ext, base, ext)
            if value in implicit_filetypes:
                return ''
        return super(SurfaceTransform, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self.inputs.out_file
        if not isdefined(outputs['out_file']):
            if isdefined(self.inputs.source_file):
                source = self.inputs.source_file
            else:
                source = self.inputs.source_annot_file
            bad_extensions = ['.%s' % e for e in ['area', 'mid', 'pial', 'avg_curv', 'curv', 'inflated', 'jacobian_white', 'orig', 'nofix', 'smoothwm', 'crv', 'sphere', 'sulc', 'thickness', 'volume', 'white']]
            use_ext = True
            if split_filename(source)[2] in bad_extensions:
                source = source + '.stripme'
                use_ext = False
            ext = ''
            if isdefined(self.inputs.target_type):
                ext = '.' + filemap[self.inputs.target_type]
                use_ext = False
            outputs['out_file'] = fname_presuffix(source, suffix='.%s%s' % (self.inputs.target_subject, ext), newpath=os.getcwd(), use_ext=use_ext)
        else:
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]
        return None