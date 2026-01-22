from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import pkg_resources
class ResourceYAMLData(YAMLData):
    """A data holder object for data parsed from a resources.yaml file."""

    @classmethod
    def FromPath(cls, resource_path):
        """Constructs a ResourceYAMLData from a standard resource_path.

    Args:
      resource_path: string, the dotted path of the resources.yaml file, e.g.
        iot.device or compute.instance.

    Returns:
      A ResourceYAMLData object.

    Raises:
      InvalidResourcePathError: invalid resource_path string.
    """
        match = re.search(_RESOURCE_PATH_PATTERN, resource_path)
        if not match:
            raise InvalidResourcePathError('Invalid resource_path: [{}].'.format(resource_path))
        surface_name = match.group('surface_name')
        resource_name = match.group('resource_name')
        dir_name = _RESOURCE_FILE_PREFIX + surface_name + '.'
        resource_file = pkg_resources.GetResource(dir_name, _RESOURCE_FILE_NAME)
        resource_data = yaml.load(resource_file)[resource_name]
        return cls(resource_data)

    def GetArgName(self):
        return self._data.get('name', None)