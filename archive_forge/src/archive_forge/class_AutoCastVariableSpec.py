import threading
from typing import Optional
import tensorflow.compat.v2 as tf
from keras.src.distribute import distributed_training_utils
class AutoCastVariableSpec(tf.types.experimental.TraceType):
    """TraceType for AutoCastVariableSpec for tracing with tf.function.

    This class implements the Type for AutoCastVariable used in tracing.
    """

    def __init__(self, value):
        self._value = value

    def is_subtype_of(self, other) -> bool:
        """If the other spec is the same as `self`, return True."""
        return self == other

    def most_specific_common_supertype(self, others):
        """`self` is the common supertype if all input types match it."""
        return self if all((self == other for other in others)) else None

    def placeholder_value(self, placeholder_context=None):
        """Use the AutoCastVariable value itself as a placeholder."""
        return self._value

    def _cast(self, value, _):
        return value

    def _to_tensors(self, value):
        return []

    def __hash__(self) -> int:
        return hash(id(self._value))

    def __eq__(self, other) -> bool:
        return self is other