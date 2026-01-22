import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class LTAConvertInputSpec(CommandLineInputSpec):
    _in_xor = ('in_lta', 'in_fsl', 'in_mni', 'in_reg', 'in_niftyreg', 'in_itk')
    in_lta = traits.Either(File(exists=True), 'identity.nofile', argstr='--inlta %s', mandatory=True, xor=_in_xor, desc='input transform of LTA type')
    in_fsl = File(exists=True, argstr='--infsl %s', mandatory=True, xor=_in_xor, desc='input transform of FSL type')
    in_mni = File(exists=True, argstr='--inmni %s', mandatory=True, xor=_in_xor, desc='input transform of MNI/XFM type')
    in_reg = File(exists=True, argstr='--inreg %s', mandatory=True, xor=_in_xor, desc='input transform of TK REG type (deprecated format)')
    in_niftyreg = File(exists=True, argstr='--inniftyreg %s', mandatory=True, xor=_in_xor, desc='input transform of Nifty Reg type (inverse RAS2RAS)')
    in_itk = File(exists=True, argstr='--initk %s', mandatory=True, xor=_in_xor, desc='input transform of ITK type')
    out_lta = traits.Either(traits.Bool, File, argstr='--outlta %s', desc='output linear transform (LTA Freesurfer format)')
    out_fsl = traits.Either(traits.Bool, File, argstr='--outfsl %s', desc='output transform in FSL format')
    out_mni = traits.Either(traits.Bool, File, argstr='--outmni %s', desc='output transform in MNI/XFM format')
    out_reg = traits.Either(traits.Bool, File, argstr='--outreg %s', desc='output transform in reg dat format')
    out_itk = traits.Either(traits.Bool, File, argstr='--outitk %s', desc='output transform in ITK format')
    invert = traits.Bool(argstr='--invert')
    ltavox2vox = traits.Bool(argstr='--ltavox2vox', requires=['out_lta'])
    source_file = File(exists=True, argstr='--src %s')
    target_file = File(exists=True, argstr='--trg %s')
    target_conform = traits.Bool(argstr='--trgconform')