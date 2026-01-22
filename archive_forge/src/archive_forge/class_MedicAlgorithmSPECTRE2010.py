import os
from ..base import (
class MedicAlgorithmSPECTRE2010(SEMLikeCommandLine):
    """SPECTRE 2010: Simple Paradigm for Extra-Cranial Tissue REmoval [1]_, [2]_.

    References
    ----------

    .. [1] A. Carass, M.B. Wheeler, J. Cuzzocreo, P.-L. Bazin, S.S. Bassett, and J.L. Prince,
           'A Joint Registration and Segmentation Approach to Skull Stripping',
           Fourth IEEE International Symposium on Biomedical Imaging (ISBI 2007), Arlington, VA,
           April 12-15, 2007.
    .. [2] A. Carass, J. Cuzzocreo, M.B. Wheeler, P.-L. Bazin, S.M. Resnick, and J.L. Prince,
           'Simple paradigm for extra-cerebral tissue removal: Algorithm and analysis',
           NeuroImage 56(4):1982-1992, 2011.

    """
    input_spec = MedicAlgorithmSPECTRE2010InputSpec
    output_spec = MedicAlgorithmSPECTRE2010OutputSpec
    _cmd = 'java edu.jhu.ece.iacl.jist.cli.run edu.jhu.ece.iacl.plugins.segmentation.skull_strip.MedicAlgorithmSPECTRE2010 '
    _outputs_filenames = {'outd0': 'outd0.nii', 'outOriginal': 'outOriginal.nii', 'outMask': 'outMask.nii', 'outSplitHalves': 'outSplitHalves.nii', 'outMidsagittal': 'outMidsagittal.nii', 'outPrior': 'outPrior.nii', 'outFANTASM': 'outFANTASM.nii', 'outSegmentation': 'outSegmentation.nii', 'outStripped': 'outStripped.nii'}
    _redirect_x = True