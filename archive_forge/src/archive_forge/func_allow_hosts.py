from . import exceptions
from . import misc
from . import normalizers
def allow_hosts(self, *hosts):
    """Require the host to be one of the provided hosts.

        .. versionadded:: 1.0

        :param hosts:
            Hosts that are allowed.
        :returns:
            The validator instance.
        :rtype:
            Validator
        """
    for host in hosts:
        self.allowed_hosts.add(normalizers.normalize_host(host))
    return self