import os
import re
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
from original dicom image by diff_unpack program and contains image
from the number of directions and number of volumes in
class HARDIMatInputSpec(CommandLineInputSpec):
    bvecs = File(exists=True, desc='b vectors file', argstr='%s', position=1, mandatory=True)
    bvals = File(exists=True, desc='b values file', mandatory=True)
    out_file = File('recon_mat.dat', desc='output matrix file', argstr='%s', usedefault=True, position=2)
    order = traits.Int(argstr='-order %s', desc='maximum order of spherical harmonics. must be even number. default is 4')
    odf_file = File(exists=True, argstr='-odf %s', desc='Filename that contains the reconstruction points on a HEMI-sphere.\nUse the pre-set 181 points by default')
    reference_file = File(exists=True, argstr='-ref %s', desc='Provide a dicom or nifti image as the reference for the program to\nfigure out the image orientation information. if no such info was\nfound in the given image header, the next 5 options -info, etc.,\nwill be used if provided. if image orientation info can be found\nin the given reference, all other 5 image orientation options will\nbe IGNORED')
    image_info = File(exists=True, argstr='-info %s', desc='specify image information file. the image info file is generated\nfrom original dicom image by diff_unpack program and contains image\norientation and other information needed for reconstruction and\ntracking. by default will look into the image folder for .info file')
    image_orientation_vectors = traits.List(traits.Float(), minlen=6, maxlen=6, desc='specify image orientation vectors. if just one argument given,\nwill treat it as filename and read the orientation vectors from\nthe file. if 6 arguments are given, will treat them as 6 float\nnumbers and construct the 1st and 2nd vector and calculate the 3rd\none automatically.\nthis information will be used to determine image orientation,\nas well as to adjust gradient vectors with oblique angle when', argstr='-iop %f')
    oblique_correction = traits.Bool(desc='when oblique angle(s) applied, some SIEMENS dti protocols do not\nadjust gradient accordingly, thus it requires adjustment for correct\ndiffusion tensor calculation', argstr='-oc')