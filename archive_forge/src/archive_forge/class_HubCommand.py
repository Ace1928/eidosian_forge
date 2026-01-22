from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.container.fleet import client
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
class HubCommand(object):
    """HubCommand is a mixin adding common utils to Hub commands."""

    @property
    def hubclient(self):
        """The HubClient for the current release track."""
        if not hasattr(self, '_client'):
            self._client = client.HubClient(self.ReleaseTrack())
        return self._client

    @property
    def messages(self):
        """Convenience property for hubclient.messages."""
        return self.hubclient.messages

    @staticmethod
    def Project(number=False):
        """Simple helper for getting the current project.

    Args:
      number: Boolean, whether to return the project number instead of the ID.

    Returns:
      The project ID or project number, as a string.
    """
        project = properties.VALUES.core.project.GetOrFail()
        if number:
            return project_util.GetProjectNumber(project)
        return project

    @staticmethod
    def LocationResourceName(location='global', use_number=False):
        return util.LocationResourceName(HubCommand.Project(use_number), location=location)

    @staticmethod
    def FeatureResourceName(name, project=None, location='global', use_number=False):
        """Builds the full resource name, using the core project property if no project is specified.."""
        project = project or HubCommand.Project(use_number)
        return util.FeatureResourceName(project, name, location=location)

    @staticmethod
    def MembershipResourceName(name, location='global', use_number=False):
        """Builds a full Membership name, using the core project property."""
        return util.MembershipResourceName(HubCommand.Project(use_number), name, location=location)

    @staticmethod
    def WorkspaceResourceName(name, location='global', use_number=False):
        """Builds a full Workspace name, using the core project property."""
        return util.WorkspaceResourceName(HubCommand.Project(use_number), name, location=location)

    @staticmethod
    def ScopeResourceName(name, location='global', use_number=False):
        """Builds a full Scope name, using the core project property."""
        return util.ScopeResourceName(HubCommand.Project(use_number), name, location=location)

    def WaitForHubOp(self, poller, op, message=None, warnings=True, **kwargs):
        """Helper wrapping waiter.WaitFor() with additional warning handling."""
        op_ref = self.hubclient.OperationRef(op)
        result = waiter.WaitFor(poller, op_ref, message=message, **kwargs)
        if warnings:
            final_op = poller.Poll(op_ref)
            metadata_dict = encoding.MessageToPyValue(final_op.metadata)
            if 'statusDetail' in metadata_dict:
                log.warning(metadata_dict['statusDetail'])
        return result