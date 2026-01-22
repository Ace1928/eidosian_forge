from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import os
import re
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.updater import installers
from googlecloudsdk.core.updater import schemas
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import requests
import six
class ComponentDiff(object):
    """Encapsulates the difference for a single component between snapshots.

  Attributes:
    id: str, The component id.
    name: str, The display name of the component.
    current: schemas.Component, The current component definition.
    latest: schemas.Component, The latest component definition that we can move
      to.
    state: ComponentState constant, The type of difference that exists for this
      component between the given snapshots.
  """

    def __init__(self, component_id, current_snapshot, latest_snapshot, platform_filter=None):
        """Create a new diff.

    Args:
      component_id: str, The id of this component.
      current_snapshot: ComponentSnapshot, The base snapshot to compare against.
      latest_snapshot: ComponentSnapshot, The new snapshot.
      platform_filter: platforms.Platform, A platform that components must
        match in order to be considered for any operations.
    """
        self.id = component_id
        self.__current = current_snapshot.ComponentFromId(component_id)
        self.__latest = latest_snapshot.ComponentFromId(component_id)
        self.current_version_string = self.__current.version.version_string if self.__current else None
        self.latest_version_string = self.__latest.version.version_string if self.__latest else None
        data_provider = self.__latest if self.__latest else self.__current
        self.name = data_provider.details.display_name
        self.is_hidden = data_provider.is_hidden
        self.is_configuration = data_provider.is_configuration
        self.platform_required = data_provider.platform_required
        self.state = self._ComputeState()
        self.platform = platform_filter
        active_snapshot = latest_snapshot if self.__latest else current_snapshot
        self.size = active_snapshot.GetEffectiveComponentSize(component_id, platform_filter=platform_filter)

    def _ComputeState(self):
        """Returns the component state."""
        if self.__current is None:
            return ComponentState.NEW
        elif self.__latest is None:
            return ComponentState.REMOVED
        elif self.__latest.version.build_number > self.__current.version.build_number:
            return ComponentState.UPDATE_AVAILABLE
        elif self.__latest.version.build_number < self.__current.version.build_number:
            if self.__latest.data is None and self.__current.data is None:
                return ComponentState.UP_TO_DATE
            elif bool(self.__latest.data) ^ bool(self.__current.data):
                return ComponentState.UPDATE_AVAILABLE
            elif self.__latest.data.contents_checksum != self.__current.data.contents_checksum:
                return ComponentState.UPDATE_AVAILABLE
        return ComponentState.UP_TO_DATE

    def __str__(self):
        return '[ {status} ]\t{name} ({id})\t[{current_version}]\t[{latest_version}]'.format(status=self.state.name, name=self.name, id=self.id, current_version=self.current_version_string, latest_version=self.latest_version_string)