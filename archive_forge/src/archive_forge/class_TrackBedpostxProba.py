import os
from ...utils.filemanip import split_filename
from ..base import (
class TrackBedpostxProba(Track):
    """
    Data from FSL's bedpostx can be imported into Camino for probabilistic tracking.
    (Use TrackBedpostxDeter for bedpostx deterministic tractography.)

    The tracking uses the files merged_th1samples.nii.gz, merged_ph1samples.nii.gz,
    ... , merged_thNsamples.nii.gz, merged_phNsamples.nii.gz where there are a
    maximum of N compartments (corresponding to each fiber population) in each
    voxel. These images contain M samples of theta and phi, the polar coordinates
    describing the "stick" for each compartment. At each iteration, a random number
    X between 1 and M is drawn and the Xth samples of theta and phi become the
    principal directions in the voxel.

    It also uses the N images mean_f1samples.nii.gz, ..., mean_fNsamples.nii.gz,
    normalized such that the sum of all compartments is 1. Compartments where the
    mean_f is less than a threshold are discarded and not used for tracking.
    The default value is 0.01. This can be changed with the min_vol_frac option.

    Example
    -------
    >>> import nipype.interfaces.camino as cam
    >>> track = cam.TrackBedpostxProba()
    >>> track.inputs.bedpostxdir = 'bedpostxout'
    >>> track.inputs.seed_file = 'seed_mask.nii'
    >>> track.inputs.iterations = 100
    >>> track.run()                  # doctest: +SKIP

    """
    input_spec = TrackBedpostxProbaInputSpec

    def __init__(self, command=None, **inputs):
        inputs['inputmodel'] = 'bedpostx'
        return super(TrackBedpostxProba, self).__init__(command, **inputs)