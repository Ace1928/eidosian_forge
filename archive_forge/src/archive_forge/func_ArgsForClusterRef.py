from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
import textwrap
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import utils as api_utils
from googlecloudsdk.api_lib.dataproc import compute_helpers
from googlecloudsdk.api_lib.dataproc import constants
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.api_lib.dataproc import storage_helpers
from googlecloudsdk.api_lib.dataproc import util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import times
import six
from stdin.
def ArgsForClusterRef(parser, dataproc, beta=False, alpha=False, include_deprecated=True, include_ttl_config=False, include_gke_platform_args=False, include_driver_pool_args=False):
    """Register flags for creating a dataproc cluster.

  Args:
    parser: The argparse.ArgParser to configure with dataproc cluster arguments.
    dataproc: Dataproc object that contains client, messages, and resources.
    beta: whether or not this is a beta command (may affect flag visibility)
    alpha: whether or not this is a alpha command (may affect flag visibility)
    include_deprecated: whether deprecated flags should be included
    include_ttl_config: whether to include Scheduled Delete(TTL) args
    include_gke_platform_args: whether to include GKE-based cluster args
    include_driver_pool_args: whether to include driver pool cluster args
  """
    labels_util.AddCreateLabelsFlags(parser)
    flags.AddTimeoutFlag(parser, default='35m')
    flags.AddZoneFlag(parser, short_flags=include_deprecated)
    flags.AddComponentFlag(parser)
    platform_group = parser.add_argument_group(mutex=True)
    gce_platform_group = platform_group.add_argument_group(help='    Compute Engine options for Dataproc clusters.\n    ')
    instances_flags.AddTagsArgs(gce_platform_group)
    gce_platform_group.add_argument('--metadata', type=arg_parsers.ArgDict(min_length=1), action='append', default=None, help='Metadata to be made available to the guest operating system running on the instances', metavar='KEY=VALUE')
    node_group = parser.add_argument_group(mutex=True)
    node_group.add_argument('--single-node', action='store_true', help='      Create a single node cluster.\n\n      A single node cluster has all master and worker components.\n      It cannot have any separate worker nodes. If this flag is not\n      specified, a cluster with separate workers is created.\n      ')
    worker_group = node_group.add_argument_group(help='Multi-node cluster flags')
    worker_group.add_argument('--num-workers', type=int, help='The number of worker nodes in the cluster. Defaults to server-specified.')
    min_workers = worker_group.add_argument_group(mutex=True)
    min_workers.add_argument('--min-num-workers', type=int, help='Minimum number of primary worker nodes to provision for cluster creation to succeed.')
    min_workers.add_argument('--min-worker-fraction', type=float, hidden=True, metavar='[0-1]', help='Minimum fraction of worker nodes required to create the cluster. If it is not met, cluster creation will fail. Must be a decimal value between 0 and 1. The number of required workers will be calcualted by ceil(min-worker-fraction * num_workers).')
    worker_group.add_argument('--secondary-worker-type', metavar='TYPE', choices=['preemptible', 'non-preemptible', 'spot'], default='preemptible', help='The type of the secondary worker group.')
    num_secondary_workers = worker_group.add_argument_group(mutex=True)
    num_secondary_workers.add_argument('--num-preemptible-workers', action=actions.DeprecationAction('--num-preemptible-workers', warn='The `--num-preemptible-workers` flag is deprecated. Use the `--num-secondary-workers` flag instead.'), type=int, hidden=True, help='The number of preemptible worker nodes in the cluster.')
    num_secondary_workers.add_argument('--num-secondary-workers', type=int, help='The number of secondary worker nodes in the cluster.')
    parser.add_argument('--master-machine-type', help='The type of machine to use for the master. Defaults to server-specified.')
    parser.add_argument('--worker-machine-type', help='The type of machine to use for workers. Defaults to server-specified.')
    parser.add_argument('--min-secondary-worker-fraction', help='Minimum fraction of secondary worker nodes required to create the cluster. If it is not met, cluster creation will fail. Must be a decimal value between 0 and 1. The number of required secondary workers is calculated by ceil(min-secondary-worker-fraction * num_secondary_workers). Defaults to 0.0001.', type=float)
    kms_resource_args.AddKmsKeyResourceArg(parser, 'cluster', name='--kms-key')
    if alpha:
        parser.add_argument('--secondary-worker-standard-capacity-base', hidden=False, type=int, help='The number of standard VMs in the Spot and Standard Mix feature.')
    parser.add_argument('--secondary-worker-machine-types', help='Types of machines with optional rank for secondary workers to use. Defaults to server-specified.eg. --secondary-worker-machine-types="type=e2-standard-8,type=t2d-standard-8,rank=0"', metavar='type=MACHINE_TYPE[,type=MACHINE_TYPE...][,rank=RANK]', type=ArgMultiValueDict(), action=arg_parsers.FlattenAction())
    image_parser = parser.add_mutually_exclusive_group()
    image_parser.add_argument('--image', metavar='IMAGE', help='The custom image used to create the cluster. It can be the image name, the image URI, or the image family URI, which selects the latest image from the family.')
    image_parser.add_argument('--image-version', metavar='VERSION', help='The image version to use for the cluster. Defaults to the latest version.')
    parser.add_argument('--bucket', help='      The Google Cloud Storage bucket to use by default to stage job\n      dependencies, miscellaneous config files, and job driver console output\n      when using this cluster.\n      ')
    parser.add_argument('--temp-bucket', help='      The Google Cloud Storage bucket to use by default to store\n      ephemeral cluster and jobs data, such as Spark and MapReduce history files.\n      ')
    netparser = gce_platform_group.add_argument_group(mutex=True)
    netparser.add_argument('--network', help='      The Compute Engine network that the VM instances of the cluster will be\n      part of. This is mutually exclusive with --subnet. If neither is\n      specified, this defaults to the "default" network.\n      ')
    netparser.add_argument('--subnet', help='      Specifies the subnet that the cluster will be part of. This is mutally\n      exclusive with --network.\n      ')
    parser.add_argument('--num-worker-local-ssds', type=int, help='The number of local SSDs to attach to each worker in a cluster.')
    parser.add_argument('--num-master-local-ssds', type=int, help='The number of local SSDs to attach to the master in a cluster.')
    secondary_worker_local_ssds = parser.add_argument_group(mutex=True)
    secondary_worker_local_ssds.add_argument('--num-preemptible-worker-local-ssds', type=int, hidden=True, action=actions.DeprecationAction('--num-preemptible-worker-local-ssds', warn='The `--num-preemptible-worker-local-ssds` flag is deprecated. Use the `--num-secondary-worker-local-ssds` flag instead.'), help='      The number of local SSDs to attach to each secondary worker in\n      a cluster.\n      ')
    secondary_worker_local_ssds.add_argument('--num-secondary-worker-local-ssds', type=int, help='      The number of local SSDs to attach to each preemptible worker in\n      a cluster.\n      ')
    parser.add_argument('--master-local-ssd-interface', help='      Interface to use to attach local SSDs to master node(s) in a cluster.\n      ')
    parser.add_argument('--worker-local-ssd-interface', help='      Interface to use to attach local SSDs to each worker in a cluster.\n      ')
    parser.add_argument('--secondary-worker-local-ssd-interface', help='      Interface to use to attach local SSDs to each secondary worker\n      in a cluster.\n      ')
    parser.add_argument('--initialization-actions', type=arg_parsers.ArgList(min_length=1), metavar='CLOUD_STORAGE_URI', help='A list of Google Cloud Storage URIs of executables to run on each node in the cluster.')
    parser.add_argument('--initialization-action-timeout', type=arg_parsers.Duration(), metavar='TIMEOUT', default='10m', help='The maximum duration of each initialization action. See $ gcloud topic datetimes for information on duration formats.')
    parser.add_argument('--num-masters', type=arg_parsers.CustomFunctionValidator(lambda n: int(n) in [1, 3], 'Number of masters must be 1 (Standard) or 3 (High Availability)', parser=arg_parsers.BoundedInt(1, 3)), help='      The number of master nodes in the cluster.\n\n      Number of Masters | Cluster Mode\n      --- | ---\n      1 | Standard\n      3 | High Availability\n      ')
    parser.add_argument('--properties', type=arg_parsers.ArgDict(), action=arg_parsers.UpdateAction, default={}, metavar='PREFIX:PROPERTY=VALUE', help='Specifies configuration properties for installed packages, such as Hadoop\nand Spark.\n\nProperties are mapped to configuration files by specifying a prefix, such as\n"core:io.serializations". The following are supported prefixes and their\nmappings:\n\nPrefix | File | Purpose of file\n--- | --- | ---\ncapacity-scheduler | capacity-scheduler.xml | Hadoop YARN Capacity Scheduler configuration\ncore | core-site.xml | Hadoop general configuration\ndistcp | distcp-default.xml | Hadoop Distributed Copy configuration\nhadoop-env | hadoop-env.sh | Hadoop specific environment variables\nhdfs | hdfs-site.xml | Hadoop HDFS configuration\nhive | hive-site.xml | Hive configuration\nmapred | mapred-site.xml | Hadoop MapReduce configuration\nmapred-env | mapred-env.sh | Hadoop MapReduce specific environment variables\npig | pig.properties | Pig configuration\nspark | spark-defaults.conf | Spark configuration\nspark-env | spark-env.sh | Spark specific environment variables\nyarn | yarn-site.xml | Hadoop YARN configuration\nyarn-env | yarn-env.sh | Hadoop YARN specific environment variables\n\nSee https://cloud.google.com/dataproc/docs/concepts/configuring-clusters/cluster-properties\nfor more information.\n\n')
    gce_platform_group.add_argument('--service-account', help='The Google Cloud IAM service account to be authenticated as.')
    gce_platform_group.add_argument('--scopes', type=arg_parsers.ArgList(min_length=1), metavar='SCOPE', help="Specifies scopes for the node instances. Multiple SCOPEs can be specified,\nseparated by commas.\nExamples:\n\n  $ {{command}} example-cluster --scopes https://www.googleapis.com/auth/bigtable.admin\n\n  $ {{command}} example-cluster --scopes sqlservice,bigquery\n\nThe following *minimum scopes* are necessary for the cluster to function\nproperly and are always added, even if not explicitly specified:\n\n  {minimum_scopes}\n\nIf the `--scopes` flag is not specified, the following *default scopes*\nare also included:\n\n  {additional_scopes}\n\nIf you want to enable all scopes use the 'cloud-platform' scope.\n\n{scopes_help}\n".format(minimum_scopes='\n  '.join(constants.MINIMUM_SCOPE_URIS), additional_scopes='\n  '.join(constants.ADDITIONAL_DEFAULT_SCOPE_URIS), scopes_help=compute_helpers.SCOPES_HELP))
    if include_deprecated:
        _AddDiskArgsDeprecated(parser, include_driver_pool_args)
    else:
        _AddDiskArgs(parser, include_driver_pool_args)
    ip_address_parser = parser.add_mutually_exclusive_group()
    ip_address_parser.add_argument('--no-address', action='store_true', help='      If provided, the instances in the cluster will not be assigned external\n      IP addresses.\n\n      If omitted, then the Dataproc service will apply a default policy to determine if each instance in the cluster gets an external IP address or not.\n\n      Note: Dataproc VMs need access to the Dataproc API. This can be achieved\n      without external IP addresses using Private Google Access\n      (https://cloud.google.com/compute/docs/private-google-access).\n      ')
    ip_address_parser.add_argument('--public-ip-address', action='store_true', help='      If provided, cluster instances are assigned external IP addresses.\n\n      If omitted, the Dataproc service applies a default policy to determine\n      whether or not each instance in the cluster gets an external IP address.\n\n      Note: Dataproc VMs need access to the Dataproc API. This can be achieved\n      without external IP addresses using Private Google Access\n      (https://cloud.google.com/compute/docs/private-google-access).\n      ')
    parser.add_argument('--private-ipv6-google-access-type', choices=['inherit-subnetwork', 'outbound', 'bidirectional'], help='      The private IPv6 Google access type for the cluster.\n      ')
    boot_disk_type_detailed_help = '      The type of the boot disk. The value must be `pd-balanced`,\n      `pd-ssd`, or `pd-standard`.\n      '
    parser.add_argument('--master-boot-disk-type', help=boot_disk_type_detailed_help)
    parser.add_argument('--worker-boot-disk-type', help=boot_disk_type_detailed_help)
    secondary_worker_boot_disk_type = parser.add_argument_group(mutex=True)
    secondary_worker_boot_disk_type.add_argument('--preemptible-worker-boot-disk-type', help=boot_disk_type_detailed_help, hidden=True, action=actions.DeprecationAction('--preemptible-worker-boot-disk-type', warn='The `--preemptible-worker-boot-disk-type` flag is deprecated. Use the `--secondary-worker-boot-disk-type` flag instead.'))
    secondary_worker_boot_disk_type.add_argument('--secondary-worker-boot-disk-type', help=boot_disk_type_detailed_help)
    if include_driver_pool_args:
        flags.AddDriverPoolId(parser)
        parser.add_argument('--driver-pool-boot-disk-type', help=boot_disk_type_detailed_help)
        parser.add_argument('--driver-pool-size', type=int, help='The size of the cluster driver pool.')
        parser.add_argument('--driver-pool-machine-type', help='The type of machine to use for the cluster driver pool nodes. Defaults to server-specified.')
        parser.add_argument('--num-driver-pool-local-ssds', type=int, help='        The number of local SSDs to attach to each cluster driver pool node.\n        ')
        parser.add_argument('--driver-pool-local-ssd-interface', help='        Interface to use to attach local SSDs to cluster driver pool node(s).\n        ')
    parser.add_argument('--enable-component-gateway', action='store_true', help='        Enable access to the web UIs of selected components on the cluster\n        through the component gateway.\n        ')
    parser.add_argument('--node-group', help='        The name of the sole-tenant node group to create the cluster on. Can be\n        a short name ("node-group-name") or in the format\n        "projects/{project-id}/zones/{zone}/nodeGroups/{node-group-name}".\n        ')
    parser.add_argument('--shielded-secure-boot', action='store_true', help="        The cluster's VMs will boot with secure boot enabled.\n        ")
    parser.add_argument('--shielded-vtpm', action='store_true', help="        The cluster's VMs will boot with the TPM (Trusted Platform Module) enabled.\n        A TPM is a hardware module that can be used for different security\n        operations, such as remote attestation, encryption, and sealing of keys.\n        ")
    parser.add_argument('--shielded-integrity-monitoring', action='store_true', help="        Enables monitoring and attestation of the boot integrity of the\n        cluster's VMs. vTPM (virtual Trusted Platform Module) must also be\n        enabled. A TPM is a hardware module that can be used for different\n        security operations, such as remote attestation, encryption, and sealing\n        of keys.\n        ")
    if not beta:
        parser.add_argument('--confidential-compute', action='store_true', help='        Enables Confidential VM. See https://cloud.google.com/compute/confidential-vm/docs for more information.\n        Note that Confidential VM can only be enabled when the machine types\n        are N2D (https://cloud.google.com/compute/docs/machine-types#n2d_machine_types)\n        and the image is SEV Compatible.\n        ')
    parser.add_argument('--dataproc-metastore', help='      Specify the name of a Dataproc Metastore service to be used as an\n      external metastore in the format:\n      "projects/{project-id}/locations/{region}/services/{service-name}".\n      ')
    parser.add_argument('--enable-node-groups', hidden=True, help='      Create cluster nodes using Dataproc NodeGroups. All the required VMs will be created using GCE MIG.\n      ', type=bool)
    autoscaling_group = parser.add_argument_group()
    flags.AddAutoscalingPolicyResourceArgForCluster(autoscaling_group, api_version='v1')
    if include_ttl_config:
        parser.add_argument('--max-idle', type=arg_parsers.Duration(), help='          The duration before cluster is auto-deleted after last job completes,\n          such as "2h" or "1d".\n          See $ gcloud topic datetimes for information on duration formats.\n          ')
        auto_delete_group = parser.add_mutually_exclusive_group()
        auto_delete_group.add_argument('--max-age', type=arg_parsers.Duration(), help='          The lifespan of the cluster before it is auto-deleted, such as\n          "2h" or "1d".\n          See $ gcloud topic datetimes for information on duration formats.\n          ')
        auto_delete_group.add_argument('--expiration-time', type=arg_parsers.Datetime.Parse, help='          The time when cluster will be auto-deleted, such as\n          "2017-08-29T18:52:51.142Z." See $ gcloud topic datetimes for\n          information on time formats.\n          ')
    AddKerberosGroup(parser)
    if not beta:
        AddSecureMultiTenancyGroup(parser)
    flags.AddMinCpuPlatformArgs(parser, include_driver_pool_args)
    _AddAcceleratorArgs(parser, include_driver_pool_args)
    if not beta:
        _AddMetricConfigArgs(parser, dataproc)
    AddReservationAffinityGroup(gce_platform_group, group_text='Specifies the reservation for the instance.', affinity_text='The type of reservation for the instance.')
    if include_gke_platform_args:
        gke_based_cluster_group = platform_group.add_argument_group(hidden=True, help='          Options for creating a GKE-based Dataproc cluster. Specifying any of these\n          will indicate that this cluster is intended to be a GKE-based cluster.\n          These options are mutually exclusive with GCE-based options.\n          ')
        gke_based_cluster_group.add_argument('--gke-cluster', hidden=True, help='            Required for GKE-based clusters. Specify the name of the GKE cluster to\n            deploy this GKE-based Dataproc cluster to. This should be the short name\n            and not the full path name.\n            ')
        gke_based_cluster_group.add_argument('--gke-cluster-namespace', hidden=True, help='            Optional. Specify the name of the namespace to deploy Dataproc system\n            components into. This namespace does not need to already exist.\n            ')