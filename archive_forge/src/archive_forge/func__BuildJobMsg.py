from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from apitools.base.protorpclite.messages import DecodeError
from apitools.base.py import encoding
from googlecloudsdk.api_lib.batch import jobs
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.batch import resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def _BuildJobMsg(args, job_msg, batch_msgs, release_track):
    """Build the job API message from the args.

  Args:
    args: the args from the parser.
    job_msg: the output job message.
    batch_msgs: the related version of the batch message.
    release_track: the release track from which _BuildJobMsg was called.
  """
    if job_msg.taskGroups is None:
        job_msg.taskGroups = []
    if not job_msg.taskGroups:
        job_msg.taskGroups.insert(0, batch_msgs.TaskGroup(taskSpec=batch_msgs.TaskSpec(runnables=[])))
    if args.script_file_path:
        job_msg.taskGroups[0].taskSpec.runnables.insert(0, batch_msgs.Runnable(script=batch_msgs.Script(path=args.script_file_path)))
    if args.script_text:
        job_msg.taskGroups[0].taskSpec.runnables.insert(0, batch_msgs.Runnable(script=batch_msgs.Script(text=args.script_text)))
    if args.container_commands_file or args.container_image_uri or args.container_entrypoint:
        container_cmds = []
        if args.container_commands_file:
            container_cmds = files.ReadFileContents(args.container_commands_file).splitlines()
        job_msg.taskGroups[0].taskSpec.runnables.insert(0, batch_msgs.Runnable(container=batch_msgs.Container(entrypoint=args.container_entrypoint, imageUri=args.container_image_uri, commands=container_cmds)))
    if args.priority:
        job_msg.priority = args.priority
    if release_track == base.ReleaseTrack.ALPHA:
        if job_msg.allocationPolicy is None and (args.machine_type or (args.network and args.subnetwork) or args.provisioning_model):
            job_msg.allocationPolicy = batch_msgs.AllocationPolicy()
    elif job_msg.allocationPolicy is None:
        job_msg.allocationPolicy = batch_msgs.AllocationPolicy()
    if args.machine_type:
        if job_msg.allocationPolicy.instances is None:
            job_msg.allocationPolicy.instances = []
        if not job_msg.allocationPolicy.instances:
            job_msg.allocationPolicy.instances.insert(0, batch_msgs.InstancePolicyOrTemplate())
        if job_msg.allocationPolicy.instances[0].policy is None:
            job_msg.allocationPolicy.instances[0].policy = batch_msgs.InstancePolicy()
        job_msg.allocationPolicy.instances[0].policy.machineType = args.machine_type
    if args.network and args.subnetwork:
        if job_msg.allocationPolicy.network is None:
            job_msg.allocationPolicy.network = batch_msgs.NetworkPolicy(networkInterfaces=[])
        job_msg.allocationPolicy.network.networkInterfaces.insert(0, batch_msgs.NetworkInterface(network=args.network, subnetwork=args.subnetwork, noExternalIpAddress=args.no_external_ip_address))
    if args.provisioning_model:
        if job_msg.allocationPolicy.instances is None:
            job_msg.allocationPolicy.instances = []
        if not job_msg.allocationPolicy.instances:
            job_msg.allocationPolicy.instances.insert(0, batch_msgs.InstancePolicyOrTemplate())
        if job_msg.allocationPolicy.instances[0].policy is None:
            job_msg.allocationPolicy.instances[0].policy = batch_msgs.InstancePolicy()
        job_msg.allocationPolicy.instances[0].policy.provisioningModel = arg_utils.ChoiceToEnum(args.provisioning_model, batch_msgs.InstancePolicy.ProvisioningModelValueValuesEnum)