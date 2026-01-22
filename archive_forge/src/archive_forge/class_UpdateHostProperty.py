import logging
from blazarclient import command
from blazarclient import exception
class UpdateHostProperty(command.UpdatePropertyCommand):
    """Update attributes of a host property."""
    resource = 'host'
    json_indent = 4
    log = logging.getLogger(__name__ + '.UpdateHostProperty')
    name_key = 'property_name'