from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import operator
from googlecloudsdk.api_lib.run import traffic
import six
def _PercentOverride(key, spec_dict, status_targets, combined_status_targets_id):
    """Computes the optional override percent to apply to the status percent."""
    percent_override = None
    if id(status_targets) == combined_status_targets_id:
        spec_by_latest_percent = _SumPercent(spec_dict[traffic.LATEST_REVISION_KEY])
        status_percent = _SumPercent(status_targets)
        status_by_latest_percent = min(spec_by_latest_percent, status_percent)
        if key == traffic.LATEST_REVISION_KEY:
            percent_override = status_by_latest_percent
        else:
            percent_override = status_percent - status_by_latest_percent
    return percent_override