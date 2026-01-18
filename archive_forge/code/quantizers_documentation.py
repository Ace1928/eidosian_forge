from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
Returns the config of the quantizer.

        An quantizer config is a Python dictionary (serializable)
        containing all configuration parameters of the quantizer.
        The same quantizer can be reinstantiated later
        (without any saved state) from this configuration.

        This method is optional if you are just training and executing models,
        exporting to and from SavedModels, or using weight checkpoints.

        This method is required for Keras `model_to_estimator`, saving and
        loading models to HDF5 formats, Keras model cloning, some visualization
        utilities, and exporting models to and from JSON.

        Returns:
            Python dictionary.
        