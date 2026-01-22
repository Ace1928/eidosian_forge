import abc
from tensorflow.python.util.tf_export import tf_export
@abc.abstractproperty
def _is_v2_column(self):
    """Returns whether this FeatureColumn is fully conformant to the new API.

    This is needed for composition type cases where an EmbeddingColumn etc.
    might take in old categorical columns as input and then we want to use the
    old API.
    """
    pass