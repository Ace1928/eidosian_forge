from builtins import range
import os
from glob import glob
from .base import ANTSCommand, ANTSCommandInputSpec
from ..base import TraitedSpec, File, traits, isdefined, OutputMultiPath
from ...utils.filemanip import split_filename
class buildtemplateparallelInputSpec(ANTSCommandInputSpec):
    dimension = traits.Enum(3, 2, 4, argstr='-d %d', usedefault=True, desc='image dimension (2, 3 or 4)', position=1)
    out_prefix = traits.Str('antsTMPL_', argstr='-o %s', usedefault=True, desc='Prefix that is prepended to all output files (default = antsTMPL_)')
    in_files = traits.List(File(exists=True), mandatory=True, desc='list of images to generate template from', argstr='%s', position=-1)
    parallelization = traits.Enum(0, 1, 2, argstr='-c %d', usedefault=True, desc='control for parallel processing (0 = serial, 1 = use PBS, 2 = use PEXEC, 3 = use Apple XGrid')
    gradient_step_size = traits.Float(argstr='-g %f', desc='smaller magnitude results in more cautious steps (default = .25)')
    iteration_limit = traits.Int(4, argstr='-i %d', usedefault=True, desc='iterations of template construction')
    num_cores = traits.Int(argstr='-j %d', requires=['parallelization'], desc='Requires parallelization = 2 (PEXEC). Sets number of cpu cores to use')
    max_iterations = traits.List(traits.Int, argstr='-m %s', sep='x', desc='maximum number of iterations (must be list of integers in the form [J,K,L...]: J = coarsest resolution iterations, K = middle resolution iterations, L = fine resolution iterations')
    bias_field_correction = traits.Bool(argstr='-n 1', desc='Applies bias field correction to moving image')
    rigid_body_registration = traits.Bool(argstr='-r 1', desc='registers inputs before creating template (useful if no initial template available)')
    similarity_metric = traits.Enum('PR', 'CC', 'MI', 'MSQ', argstr='-s %s', desc='Type of similartiy metric used for registration (CC = cross correlation, MI = mutual information, PR = probability mapping, MSQ = mean square difference)')
    transformation_model = traits.Enum('GR', 'EL', 'SY', 'S2', 'EX', 'DD', argstr='-t %s', usedefault=True, desc='Type of transofmration model used for registration (EL = elastic transformation model, SY = SyN with time, arbitrary number of time points, S2 =  SyN with time optimized for 2 time points, GR = greedy SyN, EX = exponential, DD = diffeomorphic demons style exponential mapping')
    use_first_as_target = traits.Bool(desc='uses first volume as target of all inputs. When not used, an unbiased average image is used to start.')