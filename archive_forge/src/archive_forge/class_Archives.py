from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.projects import util
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class Archives(base.Group):
    """Manage Apigee archive deployments."""
    detailed_help = {'EXAMPLES': "\n          To deploy a local archive deployment remotely to the management plane\n          in the ``test'' environment, run:\n\n              $ {command} deploy --environment=test\n\n          To list all archive deployments in the ``dev'' environment, run:\n\n              $ {command} list --environment=dev\n\n          To describe the archive deployment with id ``abcdef01234'' in the\n          ``demo'' environment of the ``my-org'' Apigee organization, run:\n\n              $ {command} describe abcdef01234 --environment=demo --organization=my-org\n\n          To update the labels of the archive deployment with id\n          ``uvxwzy56789'' in the ``test'' environment, run:\n\n              $ {command} update uvxwzy56789 --environment=demo --update-labels=foo=1,bar=2\n      "}