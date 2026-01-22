import json
from tensorboard.data import provider
from tensorboard.plugins.debugger_v2 import debug_data_multiplexer
def alerts_run_tag_filter(run, begin, end, alert_type=None):
    """Create a RunTagFilter for Alerts.

    Args:
      run: tfdbg2 run name.
      begin: Beginning index of alerts.
      end: Ending index of alerts.
      alert_type: Optional alert type, used to restrict retrieval of alerts
        data to a single type of alerts.

    Returns:
      `RunTagFilter` for the run and range of Alerts.
    """
    tag = '%s_%d_%d' % (ALERTS_BLOB_TAG_PREFIX, begin, end)
    if alert_type is not None:
        tag += '_%s' % alert_type
    return provider.RunTagFilter(runs=[run], tags=[tag])