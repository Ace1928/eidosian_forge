from __future__ import absolute_import, division, print_function
import sys
import numpy as np
from ..audio.signal import signal_frame, smooth as smooth_signal
from ..ml.nn import average_predictions
from ..processors import (OnlineProcessor, ParallelProcessor, Processor,
class MultiModelSelectionProcessor(Processor):
    """
    Processor for selecting the most suitable model (i.e. the predictions
    thereof) from a multiple models/predictions.

    Parameters
    ----------
    num_ref_predictions : int
        Number of reference predictions (see below).

    Notes
    -----
    This processor selects the most suitable prediction from multiple models by
    comparing them to the predictions of a reference model. The one with the
    smallest mean squared error is chosen.

    If `num_ref_predictions` is 0 or None, an averaged prediction is computed
    from the given predictions and used as reference.

    References
    ----------
    .. [1] Sebastian Böck, Florian Krebs and Gerhard Widmer,
           "A Multi-Model Approach to Beat Tracking Considering Heterogeneous
           Music Styles",
           Proceedings of the 15th International Society for Music Information
           Retrieval Conference (ISMIR), 2014.

    Examples
    --------
    The MultiModelSelectionProcessor takes a list of model predictions as it's
    call argument. Thus, `ppost_processor` of `RNNBeatProcessor` hast to be set
    to 'None' in order to get the predictions of all models.

    >>> proc = RNNBeatProcessor(post_processor=None)
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.beats.RNNBeatProcessor object at 0x...>

    When passing a file through the processor, a list with predictions, one for
    each model tested, is returned.

    >>> predictions = proc('tests/data/audio/sample.wav')
    >>> predictions  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [array([0.00535, 0.00774, ..., 0.02343, 0.04931], dtype=float32),
     array([0.0022 , 0.00282, ..., 0.00825, 0.0152 ], dtype=float32),
     ...,
     array([0.005  , 0.0052 , ..., 0.00472, 0.01524], dtype=float32),
     array([0.00319, 0.0044 , ..., 0.0081 , 0.01498], dtype=float32)]

    We can feed these predictions to the MultiModelSelectionProcessor.
    Since we do not have a dedicated reference prediction (which had to be the
    first element of the list and `num_ref_predictions` set to 1), we simply
    set `num_ref_predictions` to 'None'. MultiModelSelectionProcessor averages
    all predictions to obtain a reference prediction it compares all others to.

    >>> mm_proc = MultiModelSelectionProcessor(num_ref_predictions=None)
    >>> mm_proc(predictions)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    array([0.00759, 0.00901, ..., 0.00843, 0.01834], dtype=float32)

    """

    def __init__(self, num_ref_predictions, **kwargs):
        self.num_ref_predictions = num_ref_predictions

    def process(self, predictions, **kwargs):
        """
        Selects the most appropriate predictions form the list of predictions.

        Parameters
        ----------
        predictions : list
            Predictions (beat activation functions) of multiple models.

        Returns
        -------
        numpy array
            Most suitable prediction.

        Notes
        -----
        The reference beat activation function must be the first one in the
        list of given predictions.

        """
        num_refs = self.num_ref_predictions
        if num_refs in (None, 0):
            reference = average_predictions(predictions)
        elif num_refs > 0:
            reference = average_predictions(predictions[:num_refs])
        else:
            raise ValueError('`num_ref_predictions` must be positive or None, %s given' % num_refs)
        best_error = len(reference)
        best_prediction = np.empty(0)
        for prediction in predictions[num_refs:]:
            error = np.sum((prediction - reference) ** 2.0)
            if error < best_error:
                best_prediction = prediction
                best_error = error
        return best_prediction.ravel()