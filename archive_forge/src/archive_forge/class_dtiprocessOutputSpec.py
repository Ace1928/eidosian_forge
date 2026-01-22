import os
from ...base import (
class dtiprocessOutputSpec(TraitedSpec):
    fa_output = File(desc='Fractional Anisotropy output file', exists=True)
    md_output = File(desc='Mean Diffusivity output file', exists=True)
    fa_gradient_output = File(desc='Fractional Anisotropy Gradient output file', exists=True)
    fa_gradmag_output = File(desc='Fractional Anisotropy Gradient Magnitude output file', exists=True)
    color_fa_output = File(desc='Color Fractional Anisotropy output file', exists=True)
    principal_eigenvector_output = File(desc='Principal Eigenvectors Output', exists=True)
    negative_eigenvector_output = File(desc='Negative Eigenvectors Output: create a binary image where if any of the eigen value is below zero, the voxel is set to 1, otherwise 0.', exists=True)
    frobenius_norm_output = File(desc='Frobenius Norm Output', exists=True)
    lambda1_output = File(desc='Axial Diffusivity - Lambda 1 (largest eigenvalue) output', exists=True)
    lambda2_output = File(desc='Lambda 2 (middle eigenvalue) output', exists=True)
    lambda3_output = File(desc='Lambda 3 (smallest eigenvalue) output', exists=True)
    RD_output = File(desc='RD (Radial Diffusivity 1/2*(lambda2+lambda3)) output', exists=True)
    rot_output = File(desc='Rotated tensor output file.  Must also specify the dof file.', exists=True)
    outmask = File(desc='Name of the masked tensor field.', exists=True)
    deformation_output = File(desc='Warped tensor field based on a deformation field.  This option requires the --forward,-F transformation to be specified.', exists=True)