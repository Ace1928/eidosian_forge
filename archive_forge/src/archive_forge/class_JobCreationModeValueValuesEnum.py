from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JobCreationModeValueValuesEnum(_messages.Enum):
    """Optional. If not set, jobs are always required. If set, the query
    request will follow the behavior described JobCreationMode. This feature
    is not yet available. Jobs will always be created.

    Values:
      JOB_CREATION_MODE_UNSPECIFIED: If unspecified JOB_CREATION_REQUIRED is
        the default.
      JOB_CREATION_REQUIRED: Default. Job creation is always required.
      JOB_CREATION_OPTIONAL: Job creation is optional. Returning immediate
        results is prioritized. BigQuery will automatically determine if a Job
        needs to be created. The conditions under which BigQuery can decide to
        not create a Job are subject to change. If Job creation is required,
        JOB_CREATION_REQUIRED mode should be used, which is the default.
    """
    JOB_CREATION_MODE_UNSPECIFIED = 0
    JOB_CREATION_REQUIRED = 1
    JOB_CREATION_OPTIONAL = 2