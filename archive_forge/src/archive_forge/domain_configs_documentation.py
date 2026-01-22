from keystoneclient import base
from keystoneclient import exceptions
from keystoneclient.i18n import _
Delete a config for a domain.

        :param domain: the domain which the config will be deleted on
                       the server.
        :type domain: str or :class:`keystoneclient.v3.domains.Domain`

        :returns: Response object with 204 status.
        :rtype: :class:`requests.models.Response`

        