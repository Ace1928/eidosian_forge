from . import exceptions
from . import misc
from . import normalizers
def allow_ports(self, *ports):
    """Require the port to be one of the provided ports.

        .. versionadded:: 1.0

        :param ports:
            Ports that are allowed.
        :returns:
            The validator instance.
        :rtype:
            Validator
        """
    for port in ports:
        port_int = int(port, base=10)
        if 0 <= port_int <= 65535:
            self.allowed_ports.add(port)
    return self