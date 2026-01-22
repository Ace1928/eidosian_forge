from ..base import CommandLineInputSpec, CommandLine, TraitedSpec, File
class Vnifti2Image(CommandLine):
    """
    Convert a nifti file into a vista file.

    Example
    -------
    >>> vimage = Vnifti2Image()
    >>> vimage.inputs.in_file = 'image.nii'
    >>> vimage.cmdline
    'vnifti2image -in image.nii -out image.v'
    >>> vimage.run()  # doctest: +SKIP

    """
    _cmd = 'vnifti2image'
    input_spec = Vnifti2ImageInputSpec
    output_spec = Vnifti2ImageOutputSpec