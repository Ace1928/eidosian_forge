import collections
from osc_lib.command import command
from heatclient._i18n import _
from heatclient.common import format_utils
from heatclient import exc
Print failed resources.

        If the resource is a deployment resource, look up the deployment and
        print deploy_stdout and deploy_stderr.
        