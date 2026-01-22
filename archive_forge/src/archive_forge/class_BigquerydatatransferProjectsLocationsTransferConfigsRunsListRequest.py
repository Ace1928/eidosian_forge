from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigquerydatatransferProjectsLocationsTransferConfigsRunsListRequest(_messages.Message):
    """A BigquerydatatransferProjectsLocationsTransferConfigsRunsListRequest
  object.

  Enums:
    RunAttemptValueValuesEnum: Indicates how run attempts are to be pulled.
    StatesValueValuesEnum: When specified, only transfer runs with requested
      states are returned.

  Fields:
    pageSize: Page size. The default page size is the maximum value of 1000
      results.
    pageToken: Pagination token, which can be used to request a specific page
      of `ListTransferRunsRequest` list results. For multiple-page results,
      `ListTransferRunsResponse` outputs a `next_page` token, which can be
      used as the `page_token` value to request the next page of list results.
    parent: Required. Name of transfer configuration for which transfer runs
      should be retrieved. Format of transfer configuration resource name is:
      `projects/{project_id}/transferConfigs/{config_id}` or `projects/{projec
      t_id}/locations/{location_id}/transferConfigs/{config_id}`.
    runAttempt: Indicates how run attempts are to be pulled.
    states: When specified, only transfer runs with requested states are
      returned.
  """

    class RunAttemptValueValuesEnum(_messages.Enum):
        """Indicates how run attempts are to be pulled.

    Values:
      RUN_ATTEMPT_UNSPECIFIED: All runs should be returned.
      LATEST: Only latest run per day should be returned.
    """
        RUN_ATTEMPT_UNSPECIFIED = 0
        LATEST = 1

    class StatesValueValuesEnum(_messages.Enum):
        """When specified, only transfer runs with requested states are returned.

    Values:
      TRANSFER_STATE_UNSPECIFIED: State placeholder (0).
      PENDING: Data transfer is scheduled and is waiting to be picked up by
        data transfer backend (2).
      RUNNING: Data transfer is in progress (3).
      SUCCEEDED: Data transfer completed successfully (4).
      FAILED: Data transfer failed (5).
      CANCELLED: Data transfer is cancelled (6).
    """
        TRANSFER_STATE_UNSPECIFIED = 0
        PENDING = 1
        RUNNING = 2
        SUCCEEDED = 3
        FAILED = 4
        CANCELLED = 5
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    runAttempt = _messages.EnumField('RunAttemptValueValuesEnum', 4)
    states = _messages.EnumField('StatesValueValuesEnum', 5, repeated=True)