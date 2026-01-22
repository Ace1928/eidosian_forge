import collections
from osc_lib.command import command
from heatclient._i18n import _
from heatclient.common import format_utils
from heatclient import exc
def _build_failed_resources(self, stack):
    """List information about FAILED stack resources.

        Failed resources are added by recursing from the top level stack into
        failed nested stack resources. A failed nested stack resource is only
        added to the failed list if it contains no failed resources.
        """
    s = self.heat_client.stacks.get(stack)
    if s.status != 'FAILED':
        return []
    resources = self.heat_client.resources.list(s.id)
    failures = collections.OrderedDict()
    self._append_failed_resources(failures, resources, [s.stack_name])
    return failures