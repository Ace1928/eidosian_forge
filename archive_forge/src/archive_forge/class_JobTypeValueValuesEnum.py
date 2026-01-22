from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JobTypeValueValuesEnum(_messages.Enum):
    """The type of job that was executed.

    Values:
      BACKGROUND_JOB_TYPE_UNSPECIFIED: Unspecified background job type.
      BACKGROUND_JOB_TYPE_SOURCE_SEED: Job to seed from the source database.
      BACKGROUND_JOB_TYPE_CONVERT: Job to convert the source database into a
        draft of the destination database.
      BACKGROUND_JOB_TYPE_APPLY_DESTINATION: Job to apply the draft tree onto
        the destination.
      BACKGROUND_JOB_TYPE_IMPORT_RULES_FILE: Job to import and convert mapping
        rules from an external source such as an ora2pg config file.
    """
    BACKGROUND_JOB_TYPE_UNSPECIFIED = 0
    BACKGROUND_JOB_TYPE_SOURCE_SEED = 1
    BACKGROUND_JOB_TYPE_CONVERT = 2
    BACKGROUND_JOB_TYPE_APPLY_DESTINATION = 3
    BACKGROUND_JOB_TYPE_IMPORT_RULES_FILE = 4