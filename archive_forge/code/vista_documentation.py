from ..base import CommandLineInputSpec, CommandLine, TraitedSpec, File

    Convert a nifti file into a vista file.

    Example
    -------
    >>> vimage = VtoMat()
    >>> vimage.inputs.in_file = 'image.v'
    >>> vimage.cmdline
    'vtomat -in image.v -out image.mat'
    >>> vimage.run()  # doctest: +SKIP

    