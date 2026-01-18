from novaclient import api_versions
from novaclient import base
from novaclient import exceptions
from novaclient.i18n import _
Create (allocate) a server group.

        :param name: The name of the server group.
        :param policy: Policy name to associate with the server group.
        :param rules: The rules of policy which is a dict, can be applied to
            the policy, now only ``max_server_per_host`` for ``anti-affinity``
            policy would be supported (optional).
        :rtype: list of :class:`ServerGroup`
        