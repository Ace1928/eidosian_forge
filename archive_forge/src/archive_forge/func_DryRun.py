from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py.exceptions import HttpConflictError
from apitools.base.py.exceptions import HttpNotFoundError
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute.tpus import flags as tpus_flags
from googlecloudsdk.command_lib.compute.tpus.execution_groups import util as tpu_utils
from googlecloudsdk.core import log
def DryRun(self, args):
    if not args.vm_only:
        log.status.Print('Creating TPU with Name:{}, Accelerator type:{}, TF version:{}, Zone:{}, Network:{}'.format(args.name, args.accelerator_type, args.tf_version, args.zone, args.network))
        log.status.Print('Adding Storage and Logging access on TPU Service Account')
    if not args.tpu_only:
        log.status.Print('Creating VM with Name:{}, Zone:{}, Machine Type:{}, Disk Size(GB):{}, Preemptible:{}, Network:{}'.format(args.name, args.zone, args.machine_type, utils.BytesToGb(args.disk_size), args.preemptible_vm, args.network))
        log.status.Print('SSH to VM:{}'.format(args.name))