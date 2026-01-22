import os
from ...utils.filemanip import ensure_list
from ..base import TraitedSpec, File, Str, traits, InputMultiPath, isdefined
from .base import ANTSCommand, ANTSCommandInputSpec, LOCAL_DEFAULT_NUMBER_OF_THREADS
class CompositeTransformUtil(ANTSCommand):
    """
    ANTs utility which can combine or break apart transform files into their individual
    constituent components.

    Examples
    --------

    >>> from nipype.interfaces.ants import CompositeTransformUtil
    >>> tran = CompositeTransformUtil()
    >>> tran.inputs.process = 'disassemble'
    >>> tran.inputs.in_file = 'output_Composite.h5'
    >>> tran.cmdline
    'CompositeTransformUtil --disassemble output_Composite.h5 transform'
    >>> tran.run()  # doctest: +SKIP

    example for assembling transformation files

    >>> from nipype.interfaces.ants import CompositeTransformUtil
    >>> tran = CompositeTransformUtil()
    >>> tran.inputs.process = 'assemble'
    >>> tran.inputs.out_file = 'my.h5'
    >>> tran.inputs.in_file = ['AffineTransform.mat', 'DisplacementFieldTransform.nii.gz']
    >>> tran.cmdline
    'CompositeTransformUtil --assemble my.h5 AffineTransform.mat DisplacementFieldTransform.nii.gz '
    >>> tran.run()  # doctest: +SKIP
    """
    _cmd = 'CompositeTransformUtil'
    input_spec = CompositeTransformUtilInputSpec
    output_spec = CompositeTransformUtilOutputSpec

    def _num_threads_update(self):
        """
        CompositeTransformUtil ignores environment variables,
        so override environment update from ANTSCommand class
        """
        pass

    def _format_arg(self, name, spec, value):
        if name == 'output_prefix' and self.inputs.process == 'assemble':
            return ''
        if name == 'out_file' and self.inputs.process == 'disassemble':
            return ''
        return super(CompositeTransformUtil, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if self.inputs.process == 'disassemble':
            outputs['affine_transform'] = os.path.abspath('00_{}_AffineTransform.mat'.format(self.inputs.output_prefix))
            outputs['displacement_field'] = os.path.abspath('01_{}_DisplacementFieldTransform.nii.gz'.format(self.inputs.output_prefix))
        if self.inputs.process == 'assemble':
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs