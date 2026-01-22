from nipype.interfaces.base import (
import os
class BRAINSResampleInputSpec(CommandLineInputSpec):
    inputVolume = File(desc='Image To Warp', exists=True, argstr='--inputVolume %s')
    referenceVolume = File(desc='Reference image used only to define the output space. If not specified, the warping is done in the same space as the image to warp.', exists=True, argstr='--referenceVolume %s')
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc='Resulting deformed image', argstr='--outputVolume %s')
    pixelType = traits.Enum('float', 'short', 'ushort', 'int', 'uint', 'uchar', 'binary', desc="Specifies the pixel type for the input/output images.  The 'binary' pixel type uses a modified algorithm whereby the image is read in as unsigned char, a signed distance map is created, signed distance map is resampled, and then a thresholded image of type unsigned char is written to disk.", argstr='--pixelType %s')
    deformationVolume = File(desc='Displacement Field to be used to warp the image', exists=True, argstr='--deformationVolume %s')
    warpTransform = File(desc='Filename for the BRAINSFit transform used in place of the deformation field', exists=True, argstr='--warpTransform %s')
    interpolationMode = traits.Enum('NearestNeighbor', 'Linear', 'ResampleInPlace', 'BSpline', 'WindowedSinc', 'Hamming', 'Cosine', 'Welch', 'Lanczos', 'Blackman', desc='Type of interpolation to be used when applying transform to moving volume.  Options are Linear, ResampleInPlace, NearestNeighbor, BSpline, or WindowedSinc', argstr='--interpolationMode %s')
    inverseTransform = traits.Bool(desc='True/False is to compute inverse of given transformation. Default is false', argstr='--inverseTransform ')
    defaultValue = traits.Float(desc='Default voxel value', argstr='--defaultValue %f')
    gridSpacing = InputMultiPath(traits.Int, desc='Add warped grid to output image to help show the deformation that occurred with specified spacing.   A spacing of 0 in a dimension indicates that grid lines should be rendered to fall exactly (i.e. do not allow displacements off that plane).  This is useful for making a 2D image of grid lines from the 3D space ', sep=',', argstr='--gridSpacing %s')
    numberOfThreads = traits.Int(desc='Explicitly specify the maximum number of threads to use.', argstr='--numberOfThreads %d')