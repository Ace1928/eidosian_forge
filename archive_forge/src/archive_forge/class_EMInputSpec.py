from ..base import TraitedSpec, File, traits, CommandLineInputSpec, InputMultiPath
from .base import NiftySegCommand
from ..niftyreg.base import get_custom_path
class EMInputSpec(CommandLineInputSpec):
    """Input Spec for EM."""
    in_file = File(argstr='-in %s', exists=True, mandatory=True, desc='Input image to segment', position=4)
    mask_file = File(argstr='-mask %s', exists=True, desc='Filename of the ROI for label fusion')
    no_prior = traits.Int(argstr='-nopriors %s', mandatory=True, desc='Number of classes to use without prior', xor=['prior_4D', 'priors'])
    prior_4D = File(argstr='-prior4D %s', exists=True, mandatory=True, desc='4D file containing the priors', xor=['no_prior', 'priors'])
    priors = InputMultiPath(argstr='%s', mandatory=True, desc='List of priors filepaths.', xor=['no_prior', 'prior_4D'])
    max_iter = traits.Int(argstr='-max_iter %s', default_value=100, usedefault=True, desc='Maximum number of iterations')
    min_iter = traits.Int(argstr='-min_iter %s', default_value=0, usedefault=True, desc='Minimum number of iterations')
    bc_order_val = traits.Int(argstr='-bc_order %s', default_value=3, usedefault=True, desc='Polynomial order for the bias field')
    mrf_beta_val = traits.Float(argstr='-mrf_beta %s', desc='Weight of the Markov Random Field')
    desc = 'Bias field correction will run only if the ratio of improvement is below bc_thresh. (default=0 [OFF])'
    bc_thresh_val = traits.Float(argstr='-bc_thresh %s', default_value=0, usedefault=True, desc=desc)
    desc = 'Amount of regularization over the diagonal of the covariance matrix [above 1]'
    reg_val = traits.Float(argstr='-reg %s', desc=desc)
    desc = 'Outlier detection as in (Van Leemput TMI 2003). <fl1> is the Mahalanobis threshold [recommended between 3 and 7] <fl2> is a convergence ratio below which the outlier detection is going to be done [recommended 0.01]'
    outlier_val = traits.Tuple(traits.Float(), traits.Float(), argstr='-outlier %s %s', desc=desc)
    desc = 'Relax Priors [relaxation factor: 0<rf<1 (recommended=0.5), gaussian regularization: gstd>0 (recommended=2.0)] /only 3D/'
    relax_priors = traits.Tuple(traits.Float(), traits.Float(), argstr='-rf %s %s', desc=desc)
    out_file = File(name_source=['in_file'], name_template='%s_em.nii.gz', argstr='-out %s', desc='Output segmentation')
    out_bc_file = File(name_source=['in_file'], name_template='%s_bc_em.nii.gz', argstr='-bc_out %s', desc='Output bias corrected image')
    out_outlier_file = File(name_source=['in_file'], name_template='%s_outlier_em.nii.gz', argstr='-out_outlier %s', desc='Output outlierness image')