from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class DuplicateServiceError(exceptions.Error):
    """Two <service>.yaml files defining the same service id."""

    def __init__(self, path1, path2, service_id):
        super(DuplicateServiceError, self).__init__('[{path1}] and [{path2}] are both defining the service id [{s}]. All <service>.yaml files must have unique service ids.'.format(path1=path1, path2=path2, s=service_id))