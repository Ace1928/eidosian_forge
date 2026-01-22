import math
from keras_tuner.src.engine.hyperparameters import hp_utils
from keras_tuner.src.engine.hyperparameters import hyperparameter
def _check_sampling_arg(self):
    if self.min_value > self.max_value:
        raise ValueError(f"For HyperParameters.{self.__class__.__name__}(name='{self.name}'), min_value {str(self.min_value)} is greater than the max_value {str(self.max_value)}.")
    sampling_values = {'linear', 'log', 'reverse_log'}
    if self.sampling is None:
        self.sampling = 'linear'
    self.sampling = self.sampling.lower()
    if self.sampling not in sampling_values:
        raise ValueError(f"For HyperParameters.{self.__class__.__name__}(name='{self.name}'), sampling must be one of {sampling_values}")
    if self.sampling in {'log', 'reverse_log'} and self.min_value <= 0:
        raise ValueError(f"For HyperParameters.{self.__class__.__name__}(name='{self.name}'), sampling='{str(self.sampling)}' does not support negative values, found min_value: {str(self.min_value)}.")
    if self.sampling in {'log', 'reverse_log'} and self.step is not None and (self.step <= 1):
        raise ValueError(f"For HyperParameters.{self.__class__.__name__}(name='{self.name}'), expected step > 1 with sampling='{str(self.sampling)}'. Received: step={str(self.step)}.")