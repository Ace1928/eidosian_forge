from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.container.backup_restore import util as api_util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.export import util as export_util
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
def ProcessSelectedApplications(selected_applications):
    """Processes selected-applications flag."""
    if not selected_applications:
        raise exceptions.InvalidArgumentException('--selected-applications', 'Selected applications must not be empty.')
    message = api_util.GetMessagesModule()
    sa = message.NamespacedNames()
    try:
        for namespaced_name in selected_applications.split(','):
            namespace, name = namespaced_name.split('/')
            if not namespace:
                raise exceptions.InvalidArgumentException('--selected-applications', 'Namespace of selected application {0} is empty.'.format(namespaced_name))
            if not name:
                raise exceptions.InvalidArgumentException('--selected-applications', 'Name of selected application {0} is empty.'.format(namespaced_name))
            nn = message.NamespacedName()
            nn.name = name
            nn.namespace = namespace
            sa.namespacedNames.append(nn)
        return sa
    except ValueError:
        raise exceptions.InvalidArgumentException('--selected-applications', 'Selected applications {0} is invalid.'.format(selected_applications))