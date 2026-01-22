from __future__ import absolute_import, division, print_function
import numpy as np
from madmom.ml.hmm import ObservationModel, TransitionModel

        Compute the log densities of the observations using (a) GMM(s).

        Parameters
        ----------
        observations : numpy array
            Observations (i.e. multi-band spectral flux features).

        Returns
        -------
        numpy array, shape (N, num_gmms)
            Log densities of the observations, the columns represent the
            observation log probability densities for the individual GMMs.

        