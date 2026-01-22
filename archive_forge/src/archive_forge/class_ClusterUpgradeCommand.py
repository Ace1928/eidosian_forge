from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.command_lib.container.fleet.features import base as feature_base
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.util import times
import six
class ClusterUpgradeCommand(object):
    """Base class for Cluster Upgrade Feature commands."""

    def __init__(self, args):
        self.feature_name = CLUSTER_UPGRADE_FEATURE
        self.args = args

    @staticmethod
    def GetScopeNameWithProjectNumber(name):
        """Rebuilds scope name with project number instead of ID."""
        delimiter = '/'
        tokens = name.split(delimiter)
        if len(tokens) != 6 or tokens[0] != 'projects':
            raise exceptions.Error('{} is not a valid Scope resource name'.format(name))
        project_id = tokens[1]
        project_number = project_util.GetProjectNumber(project_id)
        tokens[1] = six.text_type(project_number)
        return delimiter.join(tokens)

    def ReleaseTrack(self):
        """Required to initialize HubClient. See calliope base class."""
        return self.args.calliope_command.ReleaseTrack()

    def IsClusterUpgradeRequest(self):
        """Checks if any Cluster Upgrade Feature related arguments are present."""
        cluster_upgrade_flags = {'upstream_scope', 'reset_upstream_scope', 'show_cluster_upgrade', 'show_linked_cluster_upgrade', 'default_upgrade_soaking', 'remove_upgrade_soaking_overrides', 'add_upgrade_soaking_override', 'upgrade_selector'}
        return any((has_value and flag in cluster_upgrade_flags for flag, has_value in self.args.__dict__.items()))