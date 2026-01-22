from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TimeoutFields(_messages.Message):
    """TimeoutFields allows granular specification of pipeline, task, and
  finally timeouts

  Fields:
    finally_: Finally sets the maximum allowed duration of this pipeline's
      finally
    pipeline: Pipeline sets the maximum allowed duration for execution of the
      entire pipeline. The sum of individual timeouts for tasks and finally
      must not exceed this value.
    tasks: Tasks sets the maximum allowed duration of this pipeline's tasks
  """
    finally_ = _messages.StringField(1)
    pipeline = _messages.StringField(2)
    tasks = _messages.StringField(3)