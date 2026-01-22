import logging
from blazarclient import command
from blazarclient import exception
class DeleteHost(command.DeleteCommand):
    """Delete a host."""
    resource = 'host'
    log = logging.getLogger(__name__ + '.DeleteHost')
    name_key = 'hypervisor_hostname'
    id_pattern = HOST_ID_PATTERN