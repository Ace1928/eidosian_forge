import abc
from keystoneauth1 import _utils as utils
from keystoneauth1 import plugin
Return a valid endpoint for a service.

        :param session: A session object that can be used for communication.
        :type session: keystoneauth1.session.Session
        :param string service_type: The type of service to lookup the endpoint
                                    for. This plugin will return None (failure)
                                    if service_type is not provided.
        :return: A valid endpoint URL or None if not available.
        :rtype: string or None
        