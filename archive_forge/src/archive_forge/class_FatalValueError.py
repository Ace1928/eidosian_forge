from keras_tuner.src.api_export import keras_tuner_export
@keras_tuner_export(['keras_tuner.errors.FatalValueError'])
class FatalValueError(FatalError, ValueError):
    """A fatal error during search to terminate the program.

    It is a subclass of `FatalError` and `ValueError`.

    It is used to terminate the KerasTuner program for errors that need
    users immediate attention. When this error is raised in a `Trial`, it will
    not be caught by KerasTuner.
    """
    pass