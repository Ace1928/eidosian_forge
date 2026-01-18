import re

    This method is registered as ofp_error_to_jsondict(type_, code) method
    into os_ken.ofproto.ofproto_v1_* modules.
    And this method returns ofp_error_msg as a json format for given
    'type' and 'code' defined in ofp_error_msg structure.

    Example::

        >>> ofproto.ofp_error_to_jsondict(4, 9)
        {'code': 'OFPBMC_BAD_PREREQ(9)', 'type': 'OFPET_BAD_MATCH(4)'}
    