import os
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import FSCommand, FSTraitedSpec
from .utils import copy2subjdir
class Label2Annot(FSCommand):
    """
    Converts a set of surface labels to an annotation file

    Examples
    --------
    >>> from nipype.interfaces.freesurfer import Label2Annot
    >>> l2a = Label2Annot()
    >>> l2a.inputs.hemisphere = 'lh'
    >>> l2a.inputs.subject_id = '10335'
    >>> l2a.inputs.in_labels = ['lh.aparc.label']
    >>> l2a.inputs.orig = 'lh.pial'
    >>> l2a.inputs.out_annot = 'test'
    >>> l2a.cmdline
    'mris_label2annot --hemi lh --l lh.aparc.label --a test --s 10335'
    """
    _cmd = 'mris_label2annot'
    input_spec = Label2AnnotInputSpec
    output_spec = Label2AnnotOutputSpec

    def run(self, **inputs):
        if self.inputs.copy_inputs:
            self.inputs.subjects_dir = os.getcwd()
            if 'subjects_dir' in inputs:
                inputs['subjects_dir'] = self.inputs.subjects_dir
            copy2subjdir(self, self.inputs.orig, folder='surf', basename='{0}.orig'.format(self.inputs.hemisphere))
        label_dir = os.path.join(self.inputs.subjects_dir, self.inputs.subject_id, 'label')
        if not os.path.isdir(label_dir):
            os.makedirs(label_dir)
        return super(Label2Annot, self).run(**inputs)

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = os.path.join(str(self.inputs.subjects_dir), str(self.inputs.subject_id), 'label', str(self.inputs.hemisphere) + '.' + str(self.inputs.out_annot) + '.annot')
        return outputs