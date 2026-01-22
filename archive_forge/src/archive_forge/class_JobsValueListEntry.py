from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JobsValueListEntry(_messages.Message):
    """A JobsValueListEntry object.

    Fields:
      configuration: [Full-projection-only] Specifies the job configuration.
      errorResult: A result object that will be present only if the job has
        failed.
      id: Unique opaque ID of the job.
      jobReference: Job reference uniquely identifying the job.
      kind: The resource type.
      state: Running state of the job. When the state is DONE, errorResult can
        be checked to determine whether the job succeeded or failed.
      statistics: [Output-only] Information about the job, including starting
        time and ending time of the job.
      status: [Full-projection-only] Describes the state of the job.
      user_email: [Full-projection-only] Email address of the user who ran the
        job.
    """
    configuration = _messages.MessageField('JobConfiguration', 1)
    errorResult = _messages.MessageField('ErrorProto', 2)
    id = _messages.StringField(3)
    jobReference = _messages.MessageField('JobReference', 4)
    kind = _messages.StringField(5, default=u'bigquery#job')
    state = _messages.StringField(6)
    statistics = _messages.MessageField('JobStatistics', 7)
    status = _messages.MessageField('JobStatus', 8)
    user_email = _messages.StringField(9)