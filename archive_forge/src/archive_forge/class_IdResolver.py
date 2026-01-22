from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
class IdResolver(object):
    """A class that can handle project id <--> application id transformations."""

    def __init__(self, app_ids=()):
        """Create a new IdResolver.

    Args:
     app_ids: A list of application ids with application id shard set. i.e.
         s~my_app or e~my_app.
    """
        resolver_map = {}
        for app_id in app_ids:
            resolver_map[self.resolve_project_id(app_id)] = app_id
        self._resolver_map = resolver_map

    def resolve_project_id(self, app_id):
        """Converts an application id to a project id.

    Args:
      app_id: The application id.
    Returns:
      The project id.
    """
        return app_id.rsplit('~')[-1]

    def resolve_app_id(self, project_id):
        """Converts a project id to an application id.

    Args:
      project_id: The project id.
    Returns:
      The application id.
    Raises:
      InvalidConversionError: if the application is unknown for the project id.
    """
        check_conversion(project_id in self._resolver_map, 'Cannot determine application id for provided project id: "%s".' % project_id)
        return self._resolver_map[project_id]