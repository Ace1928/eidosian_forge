from oslo_config import cfg
from heat.common.i18n import _
from heat.common import messaging
def _get_default_publisher():
    publisher_id = CONF.default_publisher_id
    if publisher_id is None:
        publisher_id = '%s.%s' % (SERVICE, CONF.host)
    return publisher_id