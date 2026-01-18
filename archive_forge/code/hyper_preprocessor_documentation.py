from autokeras.engine import named_hypermodel
Build the `tf.data` input preprocessor.

        # Arguments
            hp: `HyperParameters` instance. The hyperparameters for building the
                a Preprocessor.
            dataset: tf.data.Dataset.

        # Returns
            an instance of Preprocessor.
        