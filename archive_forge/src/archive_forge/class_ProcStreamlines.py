import os
import glob
from ...utils.filemanip import split_filename
from ..base import (
class ProcStreamlines(StdOutCommandLine):
    """
    Process streamline data

    This program does post-processing of streamline output from track. It can either output streamlines or connection probability maps.
     * http://web4.cs.ucl.ac.uk/research/medic/camino/pmwiki/pmwiki.php?n=Man.procstreamlines

    Examples
    --------

    >>> import nipype.interfaces.camino as cmon
    >>> proc = cmon.ProcStreamlines()
    >>> proc.inputs.in_file = 'tract_data.Bfloat'
    >>> proc.run()                  # doctest: +SKIP
    """
    _cmd = 'procstreamlines'
    input_spec = ProcStreamlinesInputSpec
    output_spec = ProcStreamlinesOutputSpec

    def _format_arg(self, name, spec, value):
        if name == 'outputroot':
            return spec.argstr % self._get_actual_outputroot(value)
        return super(ProcStreamlines, self)._format_arg(name, spec, value)

    def __init__(self, *args, **kwargs):
        super(ProcStreamlines, self).__init__(*args, **kwargs)
        self.outputroot_files = []

    def _run_interface(self, runtime):
        outputroot = self.inputs.outputroot
        if isdefined(outputroot):
            actual_outputroot = self._get_actual_outputroot(outputroot)
            base, filename, ext = split_filename(actual_outputroot)
            if not os.path.exists(base):
                os.makedirs(base)
            new_runtime = super(ProcStreamlines, self)._run_interface(runtime)
            self.outputroot_files = glob.glob(os.path.join(os.getcwd(), actual_outputroot + '*'))
            return new_runtime
        else:
            new_runtime = super(ProcStreamlines, self)._run_interface(runtime)
            return new_runtime

    def _get_actual_outputroot(self, outputroot):
        actual_outputroot = os.path.join('procstream_outfiles', outputroot)
        return actual_outputroot

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['proc'] = os.path.abspath(self._gen_outfilename())
        outputs['outputroot_files'] = self.outputroot_files
        return outputs

    def _gen_outfilename(self):
        _, name, _ = split_filename(self.inputs.in_file)
        return name + '_proc'