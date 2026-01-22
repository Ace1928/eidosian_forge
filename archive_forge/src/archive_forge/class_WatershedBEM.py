import os.path as op
import glob
from ... import logging
from ...utils.filemanip import simplify_list
from ..base import traits, File, Directory, TraitedSpec, OutputMultiPath
from ..freesurfer.base import FSCommand, FSTraitedSpec
class WatershedBEM(FSCommand):
    """Uses mne_watershed_bem to get information from dicom directories

    Examples
    --------

    >>> from nipype.interfaces.mne import WatershedBEM
    >>> bem = WatershedBEM()
    >>> bem.inputs.subject_id = 'subj1'
    >>> bem.inputs.subjects_dir = '.'
    >>> bem.cmdline
    'mne watershed_bem --overwrite --subject subj1 --volume T1'
    >>> bem.run()  # doctest: +SKIP

    """
    _cmd = 'mne watershed_bem'
    input_spec = WatershedBEMInputSpec
    output_spec = WatershedBEMOutputSpec
    _additional_metadata = ['loc', 'altkey']

    def _get_files(self, path, key, dirval, altkey=None):
        globsuffix = '*'
        globprefix = '*'
        keydir = op.join(path, dirval)
        if altkey:
            key = altkey
        globpattern = op.join(keydir, ''.join((globprefix, key, globsuffix)))
        return glob.glob(globpattern)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        subjects_dir = self.inputs.subjects_dir
        subject_path = op.join(subjects_dir, self.inputs.subject_id)
        output_traits = self._outputs()
        mesh_paths = []
        for k in list(outputs.keys()):
            if k != 'mesh_files':
                val = self._get_files(subject_path, k, output_traits.traits()[k].loc, output_traits.traits()[k].altkey)
                if val:
                    value_list = simplify_list(val)
                    if isinstance(value_list, list):
                        out_files = []
                        for value in value_list:
                            out_files.append(op.abspath(value))
                    elif isinstance(value_list, (str, bytes)):
                        out_files = op.abspath(value_list)
                    else:
                        raise TypeError
                    outputs[k] = out_files
                    if not k.rfind('surface') == -1:
                        mesh_paths.append(out_files)
        outputs['mesh_files'] = mesh_paths
        return outputs