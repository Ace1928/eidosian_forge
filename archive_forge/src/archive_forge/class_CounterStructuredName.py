from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CounterStructuredName(_messages.Message):
    """Identifies a counter within a per-job namespace. Counters whose
  structured names are the same get merged into a single value for the job.

  Enums:
    OriginValueValuesEnum: One of the standard Origins defined above.
    PortionValueValuesEnum: Portion of this counter, either key or value.

  Fields:
    componentStepName: Name of the optimized step being executed by the
      workers.
    executionStepName: Name of the stage. An execution step contains multiple
      component steps.
    inputIndex: Index of an input collection that's being read from/written to
      as a side input. The index identifies a step's side inputs starting by 1
      (e.g. the first side input has input_index 1, the third has input_index
      3). Side inputs are identified by a pair of (original_step_name,
      input_index). This field helps uniquely identify them.
    name: Counter name. Not necessarily globally-unique, but unique within the
      context of the other fields. Required.
    origin: One of the standard Origins defined above.
    originNamespace: A string containing a more specific namespace of the
      counter's origin.
    originalRequestingStepName: The step name requesting an operation, such as
      GBK. I.e. the ParDo causing a read/write from shuffle to occur, or a
      read from side inputs.
    originalStepName: System generated name of the original step in the user's
      graph, before optimization.
    portion: Portion of this counter, either key or value.
    workerId: ID of a particular worker.
  """

    class OriginValueValuesEnum(_messages.Enum):
        """One of the standard Origins defined above.

    Values:
      SYSTEM: Counter was created by the Dataflow system.
      USER: Counter was created by the user.
    """
        SYSTEM = 0
        USER = 1

    class PortionValueValuesEnum(_messages.Enum):
        """Portion of this counter, either key or value.

    Values:
      ALL: Counter portion has not been set.
      KEY: Counter reports a key.
      VALUE: Counter reports a value.
    """
        ALL = 0
        KEY = 1
        VALUE = 2
    componentStepName = _messages.StringField(1)
    executionStepName = _messages.StringField(2)
    inputIndex = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    name = _messages.StringField(4)
    origin = _messages.EnumField('OriginValueValuesEnum', 5)
    originNamespace = _messages.StringField(6)
    originalRequestingStepName = _messages.StringField(7)
    originalStepName = _messages.StringField(8)
    portion = _messages.EnumField('PortionValueValuesEnum', 9)
    workerId = _messages.StringField(10)