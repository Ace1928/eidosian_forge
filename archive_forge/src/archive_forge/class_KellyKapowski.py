import os
from glob import glob
from ...external.due import BibTeX
from ...utils.filemanip import split_filename, copyfile, which, fname_presuffix
from ..base import TraitedSpec, File, traits, InputMultiPath, OutputMultiPath, isdefined
from ..mixins import CopyHeaderInterface
from .base import ANTSCommand, ANTSCommandInputSpec
class KellyKapowski(ANTSCommand):
    """
    Nipype Interface to ANTs' KellyKapowski, also known as DiReCT.

    DiReCT is a registration based estimate of cortical thickness. It was published
    in S. R. Das, B. B. Avants, M. Grossman, and J. C. Gee, Registration based
    cortical thickness measurement, Neuroimage 2009, 45:867--879.

    Examples
    --------
    >>> from nipype.interfaces.ants.segmentation import KellyKapowski
    >>> kk = KellyKapowski()
    >>> kk.inputs.dimension = 3
    >>> kk.inputs.segmentation_image = "segmentation0.nii.gz"
    >>> kk.inputs.convergence = "[45,0.0,10]"
    >>> kk.inputs.thickness_prior_estimate = 10
    >>> kk.cmdline
    'KellyKapowski --convergence "[45,0.0,10]"
    --output "[segmentation0_cortical_thickness.nii.gz,segmentation0_warped_white_matter.nii.gz]"
    --image-dimensionality 3 --gradient-step 0.025000
    --maximum-number-of-invert-displacement-field-iterations 20 --number-of-integration-points 10
    --segmentation-image "[segmentation0.nii.gz,2,3]" --smoothing-variance 1.000000
    --smoothing-velocity-field-parameter 1.500000 --thickness-prior-estimate 10.000000'

    """
    _cmd = 'KellyKapowski'
    input_spec = KellyKapowskiInputSpec
    output_spec = KellyKapowskiOutputSpec
    _references = [{'entry': BibTeX('@book{Das2009867,\n  author={Sandhitsu R. Das and Brian B. Avants and Murray Grossman and James C. Gee},\n  title={Registration based cortical thickness measurement.},\n  journal={NeuroImage},\n  volume={45},\n  number={37},\n  pages={867--879},\n  year={2009},\n  issn={1053-8119},\n  url={http://www.sciencedirect.com/science/article/pii/S1053811908012780},\n  doi={https://doi.org/10.1016/j.neuroimage.2008.12.016}\n}'), 'description': 'The details on the implementation of DiReCT.', 'tags': ['implementation']}]

    def _parse_inputs(self, skip=None):
        if skip is None:
            skip = []
        skip += ['warped_white_matter', 'gray_matter_label', 'white_matter_label']
        return super(KellyKapowski, self)._parse_inputs(skip=skip)

    def _gen_filename(self, name):
        if name == 'cortical_thickness':
            output = self.inputs.cortical_thickness
            if not isdefined(output):
                _, name, ext = split_filename(self.inputs.segmentation_image)
                output = name + '_cortical_thickness' + ext
            return output
        if name == 'warped_white_matter':
            output = self.inputs.warped_white_matter
            if not isdefined(output):
                _, name, ext = split_filename(self.inputs.segmentation_image)
                output = name + '_warped_white_matter' + ext
            return output

    def _format_arg(self, opt, spec, val):
        if opt == 'segmentation_image':
            newval = '[{0},{1},{2}]'.format(self.inputs.segmentation_image, self.inputs.gray_matter_label, self.inputs.white_matter_label)
            return spec.argstr % newval
        if opt == 'cortical_thickness':
            ct = self._gen_filename('cortical_thickness')
            wm = self._gen_filename('warped_white_matter')
            newval = '[{},{}]'.format(ct, wm)
            return spec.argstr % newval
        return super(KellyKapowski, self)._format_arg(opt, spec, val)