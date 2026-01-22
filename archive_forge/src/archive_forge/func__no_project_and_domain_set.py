import argparse
import inspect
import sys
from oslo_config import cfg
import osprofiler
from osprofiler.cmd import commands
from osprofiler import exc
from osprofiler import opts
def _no_project_and_domain_set(self, args):
    if not (args.os_project_id or (args.os_project_name and (args.os_user_domain_name or args.os_user_domain_id)) or (args.os_tenant_id or args.os_tenant_name)):
        return True
    else:
        return False