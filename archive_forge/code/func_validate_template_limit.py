import collections
from oslo_config import cfg
from oslo_serialization import jsonutils
import yaml
from heat.common import exception
from heat.common.i18n import _
def validate_template_limit(contain_str):
    """Validate limit for the template.

    Check if the contain exceeds allowed size range.
    """
    if len(contain_str) > cfg.CONF.max_template_size:
        msg = _('Template size (%(actual_len)s bytes) exceeds maximum allowed size (%(limit)s bytes).') % {'actual_len': len(contain_str), 'limit': cfg.CONF.max_template_size}
        raise exception.RequestLimitExceeded(message=msg)