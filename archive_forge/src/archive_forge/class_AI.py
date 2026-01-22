import os
from warnings import warn
from ..base import traits, isdefined, TraitedSpec, File, Str, InputMultiObject
from ..mixins import CopyHeaderInterface
from .base import ANTSCommandInputSpec, ANTSCommand
class AI(ANTSCommand):
    """
    Calculate the optimal linear transform parameters for aligning two images.

    Examples
    --------
    >>> AI(
    ...     fixed_image='structural.nii',
    ...     moving_image='epi.nii',
    ...     metric=('Mattes', 32, 'Regular', 1),
    ... ).cmdline
    'antsAI -c [10,1e-06,10] -d 3 -m Mattes[structural.nii,epi.nii,32,Regular,1]
    -o initialization.mat -p 0 -s [20,0.12] -t Affine[0.1] -v 0'

    >>> AI(fixed_image='structural.nii',
    ...    moving_image='epi.nii',
    ...    metric=('Mattes', 32, 'Regular', 1),
    ...    search_grid=(12, (1, 1, 1)),
    ... ).cmdline
    'antsAI -c [10,1e-06,10] -d 3 -m Mattes[structural.nii,epi.nii,32,Regular,1]
    -o initialization.mat -p 0 -s [20,0.12] -g [12.0,1x1x1] -t Affine[0.1] -v 0'

    """
    _cmd = 'antsAI'
    input_spec = AIInputSpec
    output_spec = AIOuputSpec

    def _run_interface(self, runtime, correct_return_codes=(0,)):
        runtime = super(AI, self)._run_interface(runtime, correct_return_codes)
        self._output = {'output_transform': os.path.join(runtime.cwd, os.path.basename(self.inputs.output_transform))}
        return runtime

    def _format_arg(self, opt, spec, val):
        if opt == 'metric':
            val = '%s[{fixed_image},{moving_image},%d,%s,%g]' % val
            val = val.format(fixed_image=self.inputs.fixed_image, moving_image=self.inputs.moving_image)
            return spec.argstr % val
        if opt == 'search_grid':
            fmtval = '[%s,%s]' % (val[0], 'x'.join(('%g' % v for v in val[1])))
            return spec.argstr % fmtval
        if opt == 'fixed_image_mask':
            if isdefined(self.inputs.moving_image_mask):
                return spec.argstr % ('[%s,%s]' % (val, self.inputs.moving_image_mask))
        return super(AI, self)._format_arg(opt, spec, val)

    def _list_outputs(self):
        return getattr(self, '_output')