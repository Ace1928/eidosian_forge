from __future__ import absolute_import, division, print_function
import atexit
import time
import re
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.ansible_release import __version__ as ANSIBLE_VERSION
class XAPI(object):
    """Class for XAPI session management."""
    _xapi_session = None

    @classmethod
    def connect(cls, module, disconnect_atexit=True):
        """Establishes XAPI connection and returns session reference.

        If no existing session is available, establishes a new one
        and returns it, else returns existing one.

        Args:
            module: Reference to Ansible module object.
            disconnect_atexit (bool): Controls if method should
                register atexit handler to disconnect from XenServer
                on module exit (default: True).

        Returns:
            XAPI session reference.
        """
        if cls._xapi_session is not None:
            return cls._xapi_session
        hostname = module.params['hostname']
        username = module.params['username']
        password = module.params['password']
        ignore_ssl = not module.params['validate_certs']
        if hostname == 'localhost':
            cls._xapi_session = XenAPI.xapi_local()
            username = ''
            password = ''
        else:
            if not hostname.startswith('http://') and (not hostname.startswith('https://')):
                hostname = 'http://%s' % hostname
            try:
                cls._xapi_session = XenAPI.Session(hostname, ignore_ssl=ignore_ssl)
            except TypeError:
                cls._xapi_session = XenAPI.Session(hostname)
            if not password:
                password = ''
        try:
            cls._xapi_session.login_with_password(username, password, ANSIBLE_VERSION, 'Ansible')
        except XenAPI.Failure as f:
            module.fail_json(msg='Unable to log on to XenServer at %s as %s: %s' % (hostname, username, f.details))
        if disconnect_atexit:
            atexit.register(cls._xapi_session.logout)
        return cls._xapi_session