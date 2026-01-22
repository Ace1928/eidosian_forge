import abc
from tensorflow.python.util.tf_export import tf_export
@classmethod
def _from_config(cls, config, custom_objects=None, columns_by_name=None):
    raise NotImplementedError('Must be implemented in subclasses.')