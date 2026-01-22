from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkerAssignedEvent(_messages.Message):
    """An event generated after a worker VM has been assigned to run the
  pipeline.

  Fields:
    instance: The worker's instance name.
    machineType: The machine type that was assigned for the worker.
    zone: The zone the worker is running in.
  """
    instance = _messages.StringField(1)
    machineType = _messages.StringField(2)
    zone = _messages.StringField(3)