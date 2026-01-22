from __future__ import absolute_import, division, print_function
import numpy as np
from . import layers, activations
from ...processors import Processor, ParallelProcessor, SequentialProcessor
class NeuralNetworkEnsemble(SequentialProcessor):
    """
    Neural Network ensemble class.

    Parameters
    ----------
    networks : list
        List of the Neural Networks.
    ensemble_fn : function or callable, optional
        Ensemble function to be applied to the predictions of the neural
        network ensemble (default: average predictions).
    num_threads : int, optional
        Number of parallel working threads.

    Notes
    -----
    If `ensemble_fn` is set to 'None', the predictions are returned as a list
    with the same length as the number of networks given.

    Examples
    --------
    Create a NeuralNetworkEnsemble from the networks. Instead of supplying
    the neural networks as parameter, they can also be loaded from file:

    >>> from madmom.models import ONSETS_BRNN_PP
    >>> nn = NeuralNetworkEnsemble.load(ONSETS_BRNN_PP)
    >>> nn  # doctest: +ELLIPSIS
    <madmom.ml.nn.NeuralNetworkEnsemble object at 0x...>
    >>> nn(np.array([[0], [0.5], [1], [0], [1], [2], [0]]))
    ... # doctest: +NORMALIZE_WHITESPACE
    array([0.00116, 0.00213, 0.01428, 0.00729, 0.0088 , 0.21965, 0.00532])

    """

    def __init__(self, networks, ensemble_fn=average_predictions, num_threads=None, **kwargs):
        networks_processor = ParallelProcessor(networks, num_threads=num_threads)
        super(NeuralNetworkEnsemble, self).__init__((networks_processor, ensemble_fn))

    @classmethod
    def load(cls, nn_files, **kwargs):
        """
        Instantiate a new Neural Network ensemble from a list of files.

        Parameters
        ----------
        nn_files : list
            List of neural network model file names.
        kwargs : dict, optional
            Keyword arguments passed to NeuralNetworkEnsemble.

        Returns
        -------
        NeuralNetworkEnsemble
            NeuralNetworkEnsemble instance.

        """
        networks = [NeuralNetwork.load(f) for f in nn_files]
        return cls(networks, **kwargs)

    @staticmethod
    def add_arguments(parser, nn_files):
        """
        Add neural network options to an existing parser.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        nn_files : list
            Neural network model files.

        Returns
        -------
        argparse argument group
            Neural network argument parser group.

        """
        from madmom.utils import OverrideDefaultListAction
        g = parser.add_argument_group('neural network arguments')
        g.add_argument('--nn_files', action=OverrideDefaultListAction, type=str, default=nn_files, help='average the predictions of these pre-trained neural networks (multiple files can be given, one file per argument)')
        return g