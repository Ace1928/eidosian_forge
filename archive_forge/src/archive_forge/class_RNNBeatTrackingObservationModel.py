from __future__ import absolute_import, division, print_function
import numpy as np
from madmom.ml.hmm import ObservationModel, TransitionModel
class RNNBeatTrackingObservationModel(ObservationModel):
    """
    Observation model for beat tracking with a HMM.

    Parameters
    ----------
    state_space : :class:`BeatStateSpace` instance
        BeatStateSpace instance.
    observation_lambda : int
        Split one beat period into `observation_lambda` parts, the first
        representing beat states and the remaining non-beat states.

    References
    ----------
    .. [1] Sebastian Böck, Florian Krebs and Gerhard Widmer,
           "A Multi-Model Approach to Beat Tracking Considering Heterogeneous
           Music Styles",
           Proceedings of the 15th International Society for Music Information
           Retrieval Conference (ISMIR), 2014.

    """

    def __init__(self, state_space, observation_lambda):
        self.observation_lambda = observation_lambda
        pointers = np.zeros(state_space.num_states, dtype=np.uint32)
        border = 1.0 / observation_lambda
        pointers[state_space.state_positions < border] = 1
        super(RNNBeatTrackingObservationModel, self).__init__(pointers)

    def log_densities(self, observations):
        """
        Compute the log densities of the observations.

        Parameters
        ----------
        observations : numpy array, shape (N, )
            Observations (i.e. 1D beat activations of the RNN).

        Returns
        -------
        numpy array, shape (N, 2)
            Log densities of the observations, the columns represent the
            observation log probability densities for no-beats and beats.

        """
        log_densities = np.empty((len(observations), 2), dtype=np.float)
        log_densities[:, 0] = np.log((1.0 - observations) / (self.observation_lambda - 1))
        log_densities[:, 1] = np.log(observations)
        return log_densities