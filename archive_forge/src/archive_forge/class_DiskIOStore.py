import datetime
import io
import json
import os
import re
import tempfile
import threading
import warnings
import zipfile
import numpy as np
import tensorflow.compat.v2 as tf
import keras.src as keras
from keras.src import losses
from keras.src.engine import base_layer
from keras.src.optimizers import optimizer
from keras.src.saving.serialization_lib import ObjectSharingScope
from keras.src.saving.serialization_lib import deserialize_keras_object
from keras.src.saving.serialization_lib import serialize_keras_object
from keras.src.utils import generic_utils
from keras.src.utils import io_utils
class DiskIOStore:
    """Asset store backed by disk storage.

    If `archive` is specified, then `root_path` refers to the filename
    inside the archive.

    If `archive` is not specified, then `root_path` refers to the full path of
    the target directory.
    """

    def __init__(self, root_path, archive=None, mode=None):
        self.mode = mode
        self.root_path = root_path
        self.archive = archive
        self.tmp_dir = None
        if self.archive:
            self.tmp_dir = get_temp_dir()
            if self.mode == 'r':
                self.archive.extractall(path=self.tmp_dir)
            self.working_dir = tf.io.gfile.join(self.tmp_dir, self.root_path)
            if self.mode == 'w':
                tf.io.gfile.makedirs(self.working_dir)
        elif mode == 'r':
            self.working_dir = root_path
        else:
            self.tmp_dir = get_temp_dir()
            self.working_dir = tf.io.gfile.join(self.tmp_dir, self.root_path)
            tf.io.gfile.makedirs(self.working_dir)

    def make(self, path):
        if not path:
            return self.working_dir
        path = tf.io.gfile.join(self.working_dir, path)
        if not tf.io.gfile.exists(path):
            tf.io.gfile.makedirs(path)
        return path

    def get(self, path):
        if not path:
            return self.working_dir
        path = tf.io.gfile.join(self.working_dir, path)
        if tf.io.gfile.exists(path):
            return path
        return None

    def close(self):
        if self.mode == 'w' and self.archive:
            _write_to_zip_recursively(self.archive, self.working_dir, self.root_path)
        if self.tmp_dir and tf.io.gfile.exists(self.tmp_dir):
            tf.io.gfile.rmtree(self.tmp_dir)