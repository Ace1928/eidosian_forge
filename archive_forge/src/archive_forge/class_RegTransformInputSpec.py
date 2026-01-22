import os
from ..base import TraitedSpec, File, traits, isdefined
from .base import get_custom_path, NiftyRegCommand, NiftyRegCommandInputSpec
from ...utils.filemanip import split_filename
class RegTransformInputSpec(NiftyRegCommandInputSpec):
    """Input Spec for RegTransform."""
    ref1_file = File(exists=True, desc='The input reference/target image', argstr='-ref %s', position=0)
    ref2_file = File(exists=True, desc='The input second reference/target image', argstr='-ref2 %s', position=1, requires=['ref1_file'])
    def_input = File(exists=True, argstr='-def %s', position=-2, desc='Compute deformation field from transformation', xor=['disp_input', 'flow_input', 'comp_input', 'upd_s_form_input', 'inv_aff_input', 'inv_nrr_input', 'half_input', 'make_aff_input', 'aff_2_rig_input', 'flirt_2_nr_input'])
    disp_input = File(exists=True, argstr='-disp %s', position=-2, desc='Compute displacement field from transformation', xor=['def_input', 'flow_input', 'comp_input', 'upd_s_form_input', 'inv_aff_input', 'inv_nrr_input', 'half_input', 'make_aff_input', 'aff_2_rig_input', 'flirt_2_nr_input'])
    flow_input = File(exists=True, argstr='-flow %s', position=-2, desc='Compute flow field from spline SVF', xor=['def_input', 'disp_input', 'comp_input', 'upd_s_form_input', 'inv_aff_input', 'inv_nrr_input', 'half_input', 'make_aff_input', 'aff_2_rig_input', 'flirt_2_nr_input'])
    comp_input = File(exists=True, argstr='-comp %s', position=-3, desc='compose two transformations', xor=['def_input', 'disp_input', 'flow_input', 'upd_s_form_input', 'inv_aff_input', 'inv_nrr_input', 'half_input', 'make_aff_input', 'aff_2_rig_input', 'flirt_2_nr_input'], requires=['comp_input2'])
    comp_input2 = File(exists=True, argstr='%s', position=-2, desc='compose two transformations')
    desc = 'Update s-form using the affine transformation'
    upd_s_form_input = File(exists=True, argstr='-updSform %s', position=-3, desc=desc, xor=['def_input', 'disp_input', 'flow_input', 'comp_input', 'inv_aff_input', 'inv_nrr_input', 'half_input', 'make_aff_input', 'aff_2_rig_input', 'flirt_2_nr_input'], requires=['upd_s_form_input2'])
    desc = 'Update s-form using the affine transformation'
    upd_s_form_input2 = File(exists=True, argstr='%s', position=-2, desc=desc, requires=['upd_s_form_input'])
    inv_aff_input = File(exists=True, argstr='-invAff %s', position=-2, desc='Invert an affine transformation', xor=['def_input', 'disp_input', 'flow_input', 'comp_input', 'upd_s_form_input', 'inv_nrr_input', 'half_input', 'make_aff_input', 'aff_2_rig_input', 'flirt_2_nr_input'])
    inv_nrr_input = traits.Tuple(File(exists=True), File(exists=True), desc='Invert a non-linear transformation', argstr='-invNrr %s %s', position=-2, xor=['def_input', 'disp_input', 'flow_input', 'comp_input', 'upd_s_form_input', 'inv_aff_input', 'half_input', 'make_aff_input', 'aff_2_rig_input', 'flirt_2_nr_input'])
    half_input = File(exists=True, argstr='-half %s', position=-2, desc='Half way to the input transformation', xor=['def_input', 'disp_input', 'flow_input', 'comp_input', 'upd_s_form_input', 'inv_aff_input', 'inv_nrr_input', 'make_aff_input', 'aff_2_rig_input', 'flirt_2_nr_input'])
    argstr_tmp = '-makeAff %f %f %f %f %f %f %f %f %f %f %f %f'
    make_aff_input = traits.Tuple(traits.Float, traits.Float, traits.Float, traits.Float, traits.Float, traits.Float, traits.Float, traits.Float, traits.Float, traits.Float, traits.Float, traits.Float, argstr=argstr_tmp, position=-2, desc='Make an affine transformation matrix', xor=['def_input', 'disp_input', 'flow_input', 'comp_input', 'upd_s_form_input', 'inv_aff_input', 'inv_nrr_input', 'half_input', 'aff_2_rig_input', 'flirt_2_nr_input'])
    desc = 'Extract the rigid component from affine transformation'
    aff_2_rig_input = File(exists=True, argstr='-aff2rig %s', position=-2, desc=desc, xor=['def_input', 'disp_input', 'flow_input', 'comp_input', 'upd_s_form_input', 'inv_aff_input', 'inv_nrr_input', 'half_input', 'make_aff_input', 'flirt_2_nr_input'])
    desc = 'Convert a FLIRT affine transformation to niftyreg affine transformation'
    flirt_2_nr_input = traits.Tuple(File(exists=True), File(exists=True), File(exists=True), argstr='-flirtAff2NR %s %s %s', position=-2, desc=desc, xor=['def_input', 'disp_input', 'flow_input', 'comp_input', 'upd_s_form_input', 'inv_aff_input', 'inv_nrr_input', 'half_input', 'make_aff_input', 'aff_2_rig_input'])
    out_file = File(genfile=True, position=-1, argstr='%s', desc='transformation file to write')