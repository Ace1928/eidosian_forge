import os
from magnumclient.i18n import _
from osc_lib.command import command
def _get_target_uuid(cs, args):
    target = None
    if args.cluster:
        target = cs.clusters.get(args.cluster)
    return target.uuid