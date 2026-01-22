from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
class ZoneReservationsCompleter(compute_completers.ListCommandCompleter):

    def __init__(self, **kwargs):
        super(ZoneReservationsCompleter, self).__init__(collection='compute.reservations', list_command='alpha compute reservations list --uri', **kwargs)