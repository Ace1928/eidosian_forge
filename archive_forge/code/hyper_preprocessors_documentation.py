from tensorflow import keras
from autokeras import preprocessors
from autokeras.engine import hyper_preprocessor
HyperPreprocessor without Hyperparameters to tune.

    It would always return the same preprocessor. No hyperparameters to be
    tuned.

    # Arguments
        preprocessor: The Preprocessor to return when calling build.
    