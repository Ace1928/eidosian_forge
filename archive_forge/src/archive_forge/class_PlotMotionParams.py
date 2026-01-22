import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class PlotMotionParams(FSLCommand):
    """Use fsl_tsplot to plot the estimated motion parameters from a
    realignment program.


    Examples
    --------

    >>> import nipype.interfaces.fsl as fsl
    >>> plotter = fsl.PlotMotionParams()
    >>> plotter.inputs.in_file = 'functional.par'
    >>> plotter.inputs.in_source = 'fsl'
    >>> plotter.inputs.plot_type = 'rotations'
    >>> res = plotter.run() #doctest: +SKIP


    Notes
    -----

    The 'in_source' attribute determines the order of columns that are expected
    in the source file.  FSL prints motion parameters in the order rotations,
    translations, while SPM prints them in the opposite order.  This interface
    should be able to plot timecourses of motion parameters generated from
    other sources as long as they fall under one of these two patterns.  For
    more flexibility, see the :class:`fsl.PlotTimeSeries` interface.

    """
    _cmd = 'fsl_tsplot'
    input_spec = PlotMotionParamsInputSpec
    output_spec = PlotMotionParamsOutputSpec

    def _format_arg(self, name, spec, value):
        if name == 'plot_type':
            source = self.inputs.in_source
            if self.inputs.plot_type == 'displacement':
                title = "-t 'MCFLIRT estimated mean displacement (mm)'"
                labels = '-a abs,rel'
                return '%s %s' % (title, labels)
            sfdict = dict(fsl_rot=(1, 3), fsl_tra=(4, 6), spm_rot=(4, 6), spm_tra=(1, 3))
            sfstr = '--start=%d --finish=%d' % sfdict['%s_%s' % (source, value[:3])]
            titledict = dict(fsl='MCFLIRT', spm='Realign')
            unitdict = dict(rot='radians', tra='mm')
            title = "'%s estimated %s (%s)'" % (titledict[source], value, unitdict[value[:3]])
            return '-t %s %s -a x,y,z' % (title, sfstr)
        elif name == 'plot_size':
            return '-h %d -w %d' % value
        elif name == 'in_file':
            if isinstance(value, list):
                args = ','.join(value)
                return '-i %s' % args
            else:
                return '-i %s' % value
        return super(PlotMotionParams, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self._outputs().get()
        out_file = self.inputs.out_file
        if not isdefined(out_file):
            if isinstance(self.inputs.in_file, list):
                infile = self.inputs.in_file[0]
            else:
                infile = self.inputs.in_file
            plttype = dict(rot='rot', tra='trans', dis='disp')[self.inputs.plot_type[:3]]
            out_file = fname_presuffix(infile, suffix='_%s.png' % plttype, use_ext=False)
        outputs['out_file'] = os.path.abspath(out_file)
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()['out_file']
        return None