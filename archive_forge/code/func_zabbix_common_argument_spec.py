from __future__ import absolute_import, division, print_function
def zabbix_common_argument_spec():
    """
    Return a dictionary with connection options.
    The options are commonly used by most of Zabbix modules.
    """
    return dict(http_login_user=dict(type='str', required=False, default=None), http_login_password=dict(type='str', required=False, default=None, no_log=True))