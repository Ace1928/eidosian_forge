from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.core import properties
class EnvironmentCompleter(completers.ListCommandCompleter):

    def __init__(self, **kwargs):
        location_property = properties.VALUES.notebooks.location.Get(required=True)
        super(EnvironmentCompleter, self).__init__(collection=ENVIRONMENT_COLLECTION, list_command='beta notebooks environments list --location={} --uri'.format(location_property), **kwargs)