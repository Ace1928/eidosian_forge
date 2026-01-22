import os
from ...base import (
class BRAINSResizeInputSpec(CommandLineInputSpec):
    inputVolume = File(desc='Image To Scale', exists=True, argstr='--inputVolume %s')
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc='Resulting scaled image', argstr='--outputVolume %s')
    pixelType = traits.Enum('float', 'short', 'ushort', 'int', 'uint', 'uchar', 'binary', desc="Specifies the pixel type for the input/output images.  The 'binary' pixel type uses a modified algorithm whereby the image is read in as unsigned char, a signed distance map is created, signed distance map is resampled, and then a thresholded image of type unsigned char is written to disk.", argstr='--pixelType %s')
    scaleFactor = traits.Float(desc='The scale factor for the image spacing.', argstr='--scaleFactor %f')