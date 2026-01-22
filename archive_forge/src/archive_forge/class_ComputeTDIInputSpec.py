import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class ComputeTDIInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-2, desc='input tractography')
    out_file = File('tdi.mif', argstr='%s', usedefault=True, position=-1, desc='output TDI file')
    reference = File(exists=True, argstr='-template %s', desc='a referenceimage to be used as template')
    vox_size = traits.List(traits.Int, argstr='-vox %s', sep=',', desc='voxel dimensions')
    data_type = traits.Enum('float', 'unsigned int', argstr='-datatype %s', desc='specify output image data type')
    use_dec = traits.Bool(argstr='-dec', desc='perform mapping in DEC space')
    dixel = File(argstr='-dixel %s', desc='map streamlines todixels within each voxel. Directions are stored asazimuth elevation pairs.')
    max_tod = traits.Int(argstr='-tod %d', desc='generate a Track Orientation Distribution (TOD) in each voxel.')
    contrast = traits.Enum('tdi', 'length', 'invlength', 'scalar_map', 'scalar_map_conut', 'fod_amp', 'curvature', argstr='-constrast %s', desc='define the desired form of contrast for the output image')
    in_map = File(exists=True, argstr='-image %s', desc="provide thescalar image map for generating images with 'scalar_map' contrasts, or the SHs image for fod_amp")
    stat_vox = traits.Enum('sum', 'min', 'mean', 'max', argstr='-stat_vox %s', desc='define the statistic for choosing the finalvoxel intesities for a given contrast')
    stat_tck = traits.Enum('mean', 'sum', 'min', 'max', 'median', 'mean_nonzero', 'gaussian', 'ends_min', 'ends_mean', 'ends_max', 'ends_prod', argstr='-stat_tck %s', desc='define the statistic for choosing the contribution to be made by each streamline as a function of the samples taken along their lengths.')
    fwhm_tck = traits.Float(argstr='-fwhm_tck %f', desc='define the statistic for choosing the contribution to be made by each streamline as a function of the samples taken along their lengths')
    map_zero = traits.Bool(argstr='-map_zero', desc='if a streamline has zero contribution based on the contrast & statistic, typically it is not mapped; use this option to still contribute to the map even if this is the case (these non-contributing voxels can then influence the mean value in each voxel of the map)')
    upsample = traits.Int(argstr='-upsample %d', desc='upsample the tracks by some ratio using Hermite interpolation before mapping')
    precise = traits.Bool(argstr='-precise', desc='use a more precise streamline mapping strategy, that accurately quantifies the length through each voxel (these lengths are then taken into account during TWI calculation)')
    ends_only = traits.Bool(argstr='-ends_only', desc='only map the streamline endpoints to the image')
    tck_weights = File(exists=True, argstr='-tck_weights_in %s', desc='specify a text scalar file containing the streamline weights')
    nthreads = traits.Int(argstr='-nthreads %d', desc='number of threads. if zero, the number of available cpus will be used', nohash=True)