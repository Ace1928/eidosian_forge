import os
from glob import glob
from ...external.due import BibTeX
from ...utils.filemanip import split_filename, copyfile, which, fname_presuffix
from ..base import TraitedSpec, File, traits, InputMultiPath, OutputMultiPath, isdefined
from ..mixins import CopyHeaderInterface
from .base import ANTSCommand, ANTSCommandInputSpec
class CorticalThicknessInputSpec(ANTSCommandInputSpec):
    dimension = traits.Enum(3, 2, argstr='-d %d', usedefault=True, desc='image dimension (2 or 3)')
    anatomical_image = File(exists=True, argstr='-a %s', desc='Structural *intensity* image, typically T1. If more than one anatomical image is specified, subsequently specified images are used during the segmentation process. However, only the first image is used in the registration of priors. Our suggestion would be to specify the T1 as the first image.', mandatory=True)
    brain_template = File(exists=True, argstr='-e %s', desc='Anatomical *intensity* template (possibly created using a population data set with buildtemplateparallel.sh in ANTs). This template is  *not* skull-stripped.', mandatory=True)
    brain_probability_mask = File(exists=True, argstr='-m %s', desc='brain probability mask in template space', copyfile=False, mandatory=True)
    segmentation_priors = InputMultiPath(File(exists=True), argstr='-p %s', mandatory=True)
    out_prefix = traits.Str('antsCT_', argstr='-o %s', usedefault=True, desc='Prefix that is prepended to all output files')
    image_suffix = traits.Str('nii.gz', desc='any of standard ITK formats, nii.gz is default', argstr='-s %s', usedefault=True)
    t1_registration_template = File(exists=True, desc='Anatomical *intensity* template (assumed to be skull-stripped). A common case would be where this would be the same template as specified in the -e option which is not skull stripped.', argstr='-t %s', mandatory=True)
    extraction_registration_mask = File(exists=True, argstr='-f %s', desc='Mask (defined in the template space) used during registration for brain extraction.')
    keep_temporary_files = traits.Int(argstr='-k %d', desc='Keep brain extraction/segmentation warps, etc (default = 0).')
    max_iterations = traits.Int(argstr='-i %d', desc='ANTS registration max iterations (default = 100x100x70x20)')
    prior_segmentation_weight = traits.Float(argstr='-w %f', desc='Atropos spatial prior *probability* weight for the segmentation')
    segmentation_iterations = traits.Int(argstr='-n %d', desc='N4 -> Atropos -> N4 iterations during segmentation (default = 3)')
    posterior_formulation = traits.Str(argstr='-b %s', desc="Atropos posterior formulation and whether or not to use mixture model proportions. e.g 'Socrates[1]' (default) or 'Aristotle[1]'. Choose the latter if you want use the distance priors (see also the -l option for label propagation control).")
    use_floatingpoint_precision = traits.Enum(0, 1, argstr='-j %d', desc='Use floating point precision in registrations (default = 0)')
    use_random_seeding = traits.Enum(0, 1, argstr='-u %d', desc='Use random number generated from system clock in Atropos (default = 1)')
    b_spline_smoothing = traits.Bool(argstr='-v', desc='Use B-spline SyN for registrations and B-spline exponential mapping in DiReCT.')
    cortical_label_image = File(exists=True, desc='Cortical ROI labels to use as a prior for ATITH.')
    label_propagation = traits.Str(argstr='-l %s', desc="Incorporate a distance prior one the posterior formulation.  Should be of the form 'label[lambda,boundaryProbability]' where label is a value of 1,2,3,... denoting label ID.  The label probability for anything outside the current label = boundaryProbability * exp( -lambda * distanceFromBoundary ) Intuitively, smaller lambda values will increase the spatial capture range of the distance prior.  To apply to all label values, simply omit specifying the label, i.e. -l [lambda,boundaryProbability].")
    quick_registration = traits.Bool(argstr='-q 1', desc='If = 1, use antsRegistrationSyNQuick.sh as the basis for registration during brain extraction, brain segmentation, and (optional) normalization to a template. Otherwise use antsRegistrationSyN.sh (default = 0).')
    debug = traits.Bool(argstr='-z 1', desc='If > 0, runs a faster version of the script. Only for testing. Implies -u 0. Requires single thread computation for complete reproducibility.')