from oslo_utils import timeutils
from blazarclient import base
from blazarclient.i18n import _
from blazarclient import utils
def _add_lease_date(self, values, lease, key, delta_date, positive_delta):
    delta_sec = utils.from_elapsed_time_to_delta(delta_date, pos_sign=positive_delta)
    date = timeutils.parse_strtime(lease[key], utils.LEASE_DATE_FORMAT)
    values[key] = (date + delta_sec).strftime(utils.API_DATE_FORMAT)