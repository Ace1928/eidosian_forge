from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.logging import util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import times
import six
def _LogFilterForIds(log_ids, parent):
    """Constructs a log filter expression from the log_ids and parent name."""
    if not log_ids:
        return None
    log_names = ['"{0}"'.format(util.CreateLogResourceName(parent, log_id)) for log_id in log_ids]
    log_names = ' OR '.join(log_names)
    if len(log_ids) > 1:
        log_names = '(%s)' % log_names
    return 'logName=%s' % log_names