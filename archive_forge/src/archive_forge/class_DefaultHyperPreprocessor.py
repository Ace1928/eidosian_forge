from tensorflow import keras
from autokeras import preprocessors
from autokeras.engine import hyper_preprocessor
class DefaultHyperPreprocessor(hyper_preprocessor.HyperPreprocessor):
    """HyperPreprocessor without Hyperparameters to tune.

    It would always return the same preprocessor. No hyperparameters to be
    tuned.

    # Arguments
        preprocessor: The Preprocessor to return when calling build.
    """

    def __init__(self, preprocessor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preprocessor = preprocessor

    def build(self, hp, dataset):
        return self.preprocessor

    def get_config(self):
        config = super().get_config()
        config.update({'preprocessor': preprocessors.serialize(self.preprocessor)})
        return config

    @classmethod
    def from_config(cls, config):
        config['preprocessor'] = preprocessors.deserialize(config['preprocessor'])
        return super().from_config(config)