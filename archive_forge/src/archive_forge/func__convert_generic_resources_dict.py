from .. import errors
from ..constants import IS_WINDOWS_PLATFORM
from ..utils import (
def _convert_generic_resources_dict(generic_resources):
    if isinstance(generic_resources, list):
        return generic_resources
    if not isinstance(generic_resources, dict):
        raise errors.InvalidArgument('generic_resources must be a dict or a list (found {})'.format(type(generic_resources)))
    resources = []
    for kind, value in generic_resources.items():
        resource_type = None
        if isinstance(value, int):
            resource_type = 'DiscreteResourceSpec'
        elif isinstance(value, str):
            resource_type = 'NamedResourceSpec'
        else:
            raise errors.InvalidArgument('Unsupported generic resource reservation type: {}'.format({kind: value}))
        resources.append({resource_type: {'Kind': kind, 'Value': value}})
    return resources