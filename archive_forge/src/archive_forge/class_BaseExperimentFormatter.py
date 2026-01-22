import abc
import collections
import json
from tensorboard.uploader import util
class BaseExperimentFormatter:
    """Abstract base class for formatting experiment information as a string."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def format_experiment(self, experiment, experiment_url):
        """Format the information about an experiment as a representing string.

        Args:
          experiment: An `experiment_pb2.Experiment` protobuf message for the
            experiment to be formatted.
          experiment_url: The URL at which the experiment can be accessed via
            TensorBoard.

        Returns:
          A string that represents the experiment.
        """
        pass