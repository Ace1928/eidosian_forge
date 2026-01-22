import collections.abc as collections
import json
import os
import warnings
from pathlib import Path
from shutil import copytree
from typing import Any, Dict, List, Optional, Union
from huggingface_hub import ModelHubMixin, snapshot_download
from huggingface_hub.utils import (
from .constants import CONFIG_NAME
from .hf_api import HfApi
from .utils import SoftTemporaryDirectory, logging, validate_hf_hub_args
class KerasModelHubMixin(ModelHubMixin):
    """
    Implementation of [`ModelHubMixin`] to provide model Hub upload/download
    capabilities to Keras models.


    ```python
    >>> import tensorflow as tf
    >>> from huggingface_hub import KerasModelHubMixin


    >>> class MyModel(tf.keras.Model, KerasModelHubMixin):
    ...     def __init__(self, **kwargs):
    ...         super().__init__()
    ...         self.config = kwargs.pop("config", None)
    ...         self.dummy_inputs = ...
    ...         self.layer = ...

    ...     def call(self, *args):
    ...         return ...


    >>> # Initialize and compile the model as you normally would
    >>> model = MyModel()
    >>> model.compile(...)
    >>> # Build the graph by training it or passing dummy inputs
    >>> _ = model(model.dummy_inputs)
    >>> # Save model weights to local directory
    >>> model.save_pretrained("my-awesome-model")
    >>> # Push model weights to the Hub
    >>> model.push_to_hub("my-awesome-model")
    >>> # Download and initialize weights from the Hub
    >>> model = MyModel.from_pretrained("username/super-cool-model")
    ```
    """

    def _save_pretrained(self, save_directory):
        save_pretrained_keras(self, save_directory)

    @classmethod
    def _from_pretrained(cls, model_id, revision, cache_dir, force_download, proxies, resume_download, local_files_only, token, **model_kwargs):
        """Here we just call [`from_pretrained_keras`] function so both the mixin and
        functional APIs stay in sync.

                TODO - Some args above aren't used since we are calling
                snapshot_download instead of hf_hub_download.
        """
        if is_tf_available():
            import tensorflow as tf
        else:
            raise ImportError('Called a TensorFlow-specific function but could not import it.')
        cfg = model_kwargs.pop('config', None)
        if not os.path.isdir(model_id):
            storage_folder = snapshot_download(repo_id=model_id, revision=revision, cache_dir=cache_dir, library_name='keras', library_version=get_tf_version())
        else:
            storage_folder = model_id
        model = tf.keras.models.load_model(storage_folder, **model_kwargs)
        model.config = cfg
        return model