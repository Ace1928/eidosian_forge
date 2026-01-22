import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class ParcellationStats(FSCommand):
    """
    This program computes a number of anatomical properties.

    Examples
    ========
    >>> from nipype.interfaces.freesurfer import ParcellationStats
    >>> import os
    >>> parcstats = ParcellationStats()
    >>> parcstats.inputs.subject_id = '10335'
    >>> parcstats.inputs.hemisphere = 'lh'
    >>> parcstats.inputs.wm = './../mri/wm.mgz' # doctest: +SKIP
    >>> parcstats.inputs.transform = './../mri/transforms/talairach.xfm' # doctest: +SKIP
    >>> parcstats.inputs.brainmask = './../mri/brainmask.mgz' # doctest: +SKIP
    >>> parcstats.inputs.aseg = './../mri/aseg.presurf.mgz' # doctest: +SKIP
    >>> parcstats.inputs.ribbon = './../mri/ribbon.mgz' # doctest: +SKIP
    >>> parcstats.inputs.lh_pial = 'lh.pial' # doctest: +SKIP
    >>> parcstats.inputs.rh_pial = 'lh.pial' # doctest: +SKIP
    >>> parcstats.inputs.lh_white = 'lh.white' # doctest: +SKIP
    >>> parcstats.inputs.rh_white = 'rh.white' # doctest: +SKIP
    >>> parcstats.inputs.thickness = 'lh.thickness' # doctest: +SKIP
    >>> parcstats.inputs.surface = 'white'
    >>> parcstats.inputs.out_table = 'lh.test.stats'
    >>> parcstats.inputs.out_color = 'test.ctab'
    >>> parcstats.cmdline # doctest: +SKIP
    'mris_anatomical_stats -c test.ctab -f lh.test.stats 10335 lh white'
    """
    _cmd = 'mris_anatomical_stats'
    input_spec = ParcellationStatsInputSpec
    output_spec = ParcellationStatsOutputSpec

    def run(self, **inputs):
        if self.inputs.copy_inputs:
            self.inputs.subjects_dir = os.getcwd()
            if 'subjects_dir' in inputs:
                inputs['subjects_dir'] = self.inputs.subjects_dir
            copy2subjdir(self, self.inputs.lh_white, 'surf', 'lh.white')
            copy2subjdir(self, self.inputs.lh_pial, 'surf', 'lh.pial')
            copy2subjdir(self, self.inputs.rh_white, 'surf', 'rh.white')
            copy2subjdir(self, self.inputs.rh_pial, 'surf', 'rh.pial')
            copy2subjdir(self, self.inputs.wm, 'mri', 'wm.mgz')
            copy2subjdir(self, self.inputs.transform, os.path.join('mri', 'transforms'), 'talairach.xfm')
            copy2subjdir(self, self.inputs.brainmask, 'mri', 'brainmask.mgz')
            copy2subjdir(self, self.inputs.aseg, 'mri', 'aseg.presurf.mgz')
            copy2subjdir(self, self.inputs.ribbon, 'mri', 'ribbon.mgz')
            copy2subjdir(self, self.inputs.thickness, 'surf', '{0}.thickness'.format(self.inputs.hemisphere))
            if isdefined(self.inputs.cortex_label):
                copy2subjdir(self, self.inputs.cortex_label, 'label', '{0}.cortex.label'.format(self.inputs.hemisphere))
        createoutputdirs(self._list_outputs())
        return super(ParcellationStats, self).run(**inputs)

    def _gen_filename(self, name):
        if name in ['out_table', 'out_color']:
            return self._list_outputs()[name]
        return None

    def _list_outputs(self):
        outputs = self._outputs().get()
        if isdefined(self.inputs.out_table):
            outputs['out_table'] = os.path.abspath(self.inputs.out_table)
        else:
            stats_dir = os.path.join(self.inputs.subjects_dir, self.inputs.subject_id, 'stats')
            if isdefined(self.inputs.in_annotation):
                if self.inputs.surface == 'pial':
                    basename = os.path.basename(self.inputs.in_annotation).replace('.annot', '.pial.stats')
                else:
                    basename = os.path.basename(self.inputs.in_annotation).replace('.annot', '.stats')
            elif isdefined(self.inputs.in_label):
                if self.inputs.surface == 'pial':
                    basename = os.path.basename(self.inputs.in_label).replace('.label', '.pial.stats')
                else:
                    basename = os.path.basename(self.inputs.in_label).replace('.label', '.stats')
            else:
                basename = str(self.inputs.hemisphere) + '.aparc.annot.stats'
            outputs['out_table'] = os.path.join(stats_dir, basename)
        if isdefined(self.inputs.out_color):
            outputs['out_color'] = os.path.abspath(self.inputs.out_color)
        else:
            out_dir = os.path.join(self.inputs.subjects_dir, self.inputs.subject_id, 'label')
            if isdefined(self.inputs.in_annotation):
                basename = os.path.basename(self.inputs.in_annotation)
                for item in ['lh.', 'rh.', 'aparc.', 'annot']:
                    basename = basename.replace(item, '')
                annot = basename
                if 'BA' in annot:
                    outputs['out_color'] = os.path.join(out_dir, annot + 'ctab')
                else:
                    outputs['out_color'] = os.path.join(out_dir, 'aparc.annot.' + annot + 'ctab')
            else:
                outputs['out_color'] = os.path.join(out_dir, 'aparc.annot.ctab')
        return outputs