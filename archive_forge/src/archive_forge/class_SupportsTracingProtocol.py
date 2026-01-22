import abc
from typing import Any, List, Optional, Sequence, Iterator
from typing_extensions import Protocol
from typing_extensions import runtime_checkable
from tensorflow.python.types import core
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
@runtime_checkable
class SupportsTracingProtocol(Protocol):
    """A protocol allowing custom classes to control tf.function retracing."""

    @doc_controls.doc_private
    @abc.abstractmethod
    def __tf_tracing_type__(self, context: TracingContext) -> TraceType:
        """Returns the tracing type of this object.

    The tracing type is used to build the signature of a tf.function
    when traced, and to match arguments with existing signatures.
    When a Function object is called, tf.function looks at the tracing type
    of the call arguments. If an existing signature of matching type exists,
    it will be used. Otherwise, a new function is traced, and its signature
    will use the tracing type of the call arguments.

    Args:
      context: a context object created for each function call for tracking
        information about the call arguments as a whole
    Returns:
      The tracing type of this object.
    """