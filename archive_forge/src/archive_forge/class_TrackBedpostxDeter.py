import os
from ...utils.filemanip import split_filename
from ..base import (
class TrackBedpostxDeter(Track):
    """
    Data from FSL's bedpostx can be imported into Camino for deterministic tracking.
    (Use TrackBedpostxProba for bedpostx probabilistic tractography.)

    The tracking is based on the vector images dyads1.nii.gz, ... , dyadsN.nii.gz,
    where there are a maximum of N compartments (corresponding to each fiber
    population) in each voxel.

    It also uses the N images mean_f1samples.nii.gz, ..., mean_fNsamples.nii.gz,
    normalized such that the sum of all compartments is 1. Compartments where the
    mean_f is less than a threshold are discarded and not used for tracking.
    The default value is 0.01. This can be changed with the min_vol_frac option.

    Example
    -------
    >>> import nipype.interfaces.camino as cam
    >>> track = cam.TrackBedpostxDeter()
    >>> track.inputs.bedpostxdir = 'bedpostxout'
    >>> track.inputs.seed_file = 'seed_mask.nii'
    >>> track.run()                  # doctest: +SKIP

    """
    input_spec = TrackBedpostxDeterInputSpec

    def __init__(self, command=None, **inputs):
        inputs['inputmodel'] = 'bedpostx_dyad'
        return super(TrackBedpostxDeter, self).__init__(command, **inputs)