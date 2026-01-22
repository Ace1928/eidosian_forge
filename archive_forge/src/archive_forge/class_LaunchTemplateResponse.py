from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LaunchTemplateResponse(_messages.Message):
    """Response to the request to launch a template.

  Fields:
    job: The job that was launched, if the request was not a dry run and the
      job was successfully launched.
  """
    job = _messages.MessageField('Job', 1)