from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1alpha3CrawlerRun(_messages.Message):
    """A run of the crawler.

  Enums:
    RunOptionValueValuesEnum:
    StateValueValuesEnum: Output only. The state of the crawler run.

  Fields:
    error: Output only. The error status of the crawler run. This field will
      be populated only if the crawler run state is FAILED.
    name: Output only. The name of the crawler run. For example,
      "projects/project1/crawlers/crawler1/crawlerRuns/042423713e9a"
    runOption: A RunOptionValueValuesEnum attribute.
    state: Output only. The state of the crawler run.
  """

    class RunOptionValueValuesEnum(_messages.Enum):
        """RunOptionValueValuesEnum enum type.

    Values:
      RUN_OPTION_UNSPECIFIED: Unspecified run option.
      AD_HOC: Ad-hoc run option.
      SCHEDULED: Scheduled run option.
    """
        RUN_OPTION_UNSPECIFIED = 0
        AD_HOC = 1
        SCHEDULED = 2

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of the crawler run.

    Values:
      STATE_UNSPECIFIED: Unspecified crawler run state.
      PENDING: This crawler run is waiting on resources to be ready.
      RUNNING: This crawler run is running.
      FAILED: This crawler run failed.
      SUCCEEDED: This crawler run succeeded.
    """
        STATE_UNSPECIFIED = 0
        PENDING = 1
        RUNNING = 2
        FAILED = 3
        SUCCEEDED = 4
    error = _messages.MessageField('Status', 1)
    name = _messages.StringField(2)
    runOption = _messages.EnumField('RunOptionValueValuesEnum', 3)
    state = _messages.EnumField('StateValueValuesEnum', 4)