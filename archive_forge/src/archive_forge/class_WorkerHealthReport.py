from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkerHealthReport(_messages.Message):
    """WorkerHealthReport contains information about the health of a worker.
  The VM should be identified by the labels attached to the WorkerMessage that
  this health ping belongs to.

  Messages:
    PodsValueListEntry: A PodsValueListEntry object.

  Fields:
    msg: Message describing any unusual health reports.
    pods: The pods running on the worker. See:
      http://kubernetes.io/v1.1/docs/api-reference/v1/definitions.html#_v1_pod
      This field is used by the worker to send the status of the indvidual
      containers running on each worker.
    reportInterval: The interval at which the worker is sending health
      reports. The default value of 0 should be interpreted as the field is
      not being explicitly set by the worker.
    vmBrokenCode: Code to describe a specific reason, if known, that a VM has
      reported broken state.
    vmIsBroken: Whether the VM is in a permanently broken state. Broken VMs
      should be abandoned or deleted ASAP to avoid assigning or completing any
      work.
    vmIsHealthy: Whether the VM is currently healthy.
    vmStartupTime: The time the VM was booted.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PodsValueListEntry(_messages.Message):
        """A PodsValueListEntry object.

    Messages:
      AdditionalProperty: An additional property for a PodsValueListEntry
        object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a PodsValueListEntry object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    msg = _messages.StringField(1)
    pods = _messages.MessageField('PodsValueListEntry', 2, repeated=True)
    reportInterval = _messages.StringField(3)
    vmBrokenCode = _messages.StringField(4)
    vmIsBroken = _messages.BooleanField(5)
    vmIsHealthy = _messages.BooleanField(6)
    vmStartupTime = _messages.StringField(7)