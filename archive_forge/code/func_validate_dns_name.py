import re
from oslo_config import cfg
from neutron_lib._i18n import _
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib.db import constants as db_constants
def validate_dns_name(data, max_len=db_constants.FQDN_FIELD_SIZE):
    """Validate DNS name.

    This method validates dns name and also needs to have dns_domain in config
    because this may call a method which uses the config.

    :param data: The data to validate.
    :param max_len: An optional cap on the length of the string.
    :returns: None if data is valid, otherwise a human readable message
        indicating why validation failed.
    """
    msg = _validate_dns_format(data, max_len)
    if msg:
        return msg
    request_dns_name = _get_request_dns_name(data)
    if request_dns_name:
        dns_domain = _get_dns_domain_config()
        msg = _validate_dns_name_with_dns_domain(request_dns_name, dns_domain)
        if msg:
            return msg