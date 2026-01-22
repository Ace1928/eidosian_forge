from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import random
import string
import time
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py.exceptions import HttpError
from apitools.base.py.exceptions import HttpNotFoundError
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.cloudbuild import logs as cb_logs
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.services import enable_api as services_api
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions as http_exc
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.artifacts import docker_util
from googlecloudsdk.command_lib.cloudbuild import execution
from googlecloudsdk.command_lib.compute.sole_tenancy import util as sole_tenancy_util
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import requests
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding as encoding_util
import six
def RunInstanceOVFImportBuild(args, compute_client, instance_name, source_uri, no_guest_environment, can_ip_forward, deletion_protection, description, labels, machine_type, network, network_tier, subnet, private_network_ip, no_restart_on_failure, os, tags, zone, project, output_filter, release_track, hostname, no_address, byol, compute_service_account, cloudbuild_service_account, service_account, no_service_account, scopes, no_scopes, uefi_compatible):
    """Run a OVF into VM instance import build on Google Cloud Build.

  Args:
    args: an argparse namespace. All the arguments that were provided to this
      command invocation.
    compute_client: Google Compute Engine client.
    instance_name: Name of the instance to be imported.
    source_uri: A GCS path to OVA or OVF package.
    no_guest_environment: If set to True, Google Guest Environment won't be
      installed on the boot disk of the VM.
    can_ip_forward: If set to True, allows the instances to send and receive
      packets with non-matching destination or source IP addresses.
    deletion_protection: Enables deletion protection for the instance.
    description: Specifies a textual description of the instances.
    labels: List of label KEY=VALUE pairs to add to the instance.
    machine_type: Specifies the machine type used for the instances.
    network: Specifies the network that the instances will be part of.
    network_tier: Specifies the network tier of the interface. NETWORK_TIER must
      be one of: PREMIUM, STANDARD.
    subnet: Specifies the subnet that the instances will be part of.
    private_network_ip: Specifies the RFC1918 IP to assign to the instance.
    no_restart_on_failure: The instances will NOT be restarted if they are
      terminated by Compute Engine.
    os: Specifies the OS of the boot disk being imported.
    tags: A list of strings for adding tags to the Argo build.
    zone: The GCP zone to tell Daisy to do work in. If unspecified, defaults to
      wherever the Argo runner happens to be.
    project: The Google Cloud Platform project name to use for OVF import.
    output_filter: A list of strings indicating what lines from the log should
      be output. Only lines that start with one of the strings in output_filter
      will be displayed.
    release_track: release track of the command used. One of - "alpha", "beta"
      or "ga"
    hostname: hostname of the instance to be imported
    no_address: Specifies that no external IP address will be assigned to the
      instances.
    byol: Specifies that you want to import an image with an existing license.
    compute_service_account: Compute service account to be used for worker
      instances.
    cloudbuild_service_account: CloudBuild service account to be used for
      running cloud builds.
    service_account: Service account to be assigned to the VM instance or
      machine image.
    no_service_account: No service account is assigned to the VM instance or
      machine image.
    scopes: Access scopes to be assigned to the VM instance or machine image
    no_scopes: No access scopes are assigned to the VM instance or machine
      image.
    uefi_compatible: Specifies that the instance should be booted from UEFI.

  Returns:
    A build object that either streams the output or is displayed as a
    link to the build.

  Raises:
    FailedBuildException: If the build is completed and not 'SUCCESS'.
  """
    project_id = projects_util.ParseProject(properties.VALUES.core.project.GetOrFail())
    _CheckIamPermissions(project_id, frozenset(IMPORT_ROLES_FOR_CLOUDBUILD_SERVICE_ACCOUNT), frozenset(IMPORT_ROLES_FOR_COMPUTE_SERVICE_ACCOUNT), cloudbuild_service_account, compute_service_account)
    ovf_importer_args = []
    AppendArg(ovf_importer_args, 'instance-names', instance_name)
    AppendArg(ovf_importer_args, 'client-id', 'gcloud')
    AppendArg(ovf_importer_args, 'ovf-gcs-path', source_uri)
    AppendBoolArg(ovf_importer_args, 'no-guest-environment', no_guest_environment)
    AppendBoolArg(ovf_importer_args, 'can-ip-forward', can_ip_forward)
    AppendBoolArg(ovf_importer_args, 'deletion-protection', deletion_protection)
    AppendArg(ovf_importer_args, 'description', description)
    if labels:
        AppendArg(ovf_importer_args, 'labels', ','.join(['{}={}'.format(k, v) for k, v in labels.items()]))
    AppendArg(ovf_importer_args, 'machine-type', machine_type)
    AppendArg(ovf_importer_args, 'network', network)
    AppendArg(ovf_importer_args, 'network-tier', network_tier)
    AppendArg(ovf_importer_args, 'subnet', subnet)
    AppendArg(ovf_importer_args, 'private-network-ip', private_network_ip)
    AppendBoolArg(ovf_importer_args, 'no-restart-on-failure', no_restart_on_failure)
    if byol:
        AppendBoolArg(ovf_importer_args, 'byol', byol)
    if uefi_compatible:
        AppendBoolArg(ovf_importer_args, 'uefi-compatible', uefi_compatible)
    AppendArg(ovf_importer_args, 'os', os)
    if tags:
        AppendArg(ovf_importer_args, 'tags', ','.join(tags))
    AppendArg(ovf_importer_args, 'zone', zone)
    AppendArg(ovf_importer_args, 'timeout', GetDaisyTimeout(args), '-{0}={1}s')
    AppendArg(ovf_importer_args, 'project', project)
    _AppendNodeAffinityLabelArgs(ovf_importer_args, args, compute_client.messages)
    if release_track:
        AppendArg(ovf_importer_args, 'release-track', release_track)
    AppendArg(ovf_importer_args, 'hostname', hostname)
    AppendArg(ovf_importer_args, 'client-version', config.CLOUD_SDK_VERSION)
    AppendBoolArg(ovf_importer_args, 'no-external-ip', no_address)
    if compute_service_account:
        AppendArg(ovf_importer_args, 'compute-service-account', compute_service_account)
    if service_account:
        AppendArg(ovf_importer_args, 'service-account', service_account)
    elif no_service_account:
        AppendArg(ovf_importer_args, 'service-account', '', allow_empty=True)
    if scopes:
        AppendArg(ovf_importer_args, 'scopes', ','.join(scopes))
    elif no_scopes:
        AppendArg(ovf_importer_args, 'scopes', '', allow_empty=True)
    build_tags = ['gce-daisy', 'gce-ovf-import']
    backoff = lambda elapsed: 2 if elapsed < 30 else 15
    builder_region = _GetBuilderRegion(_GetInstanceImportRegion)
    builder = _GetBuilder(_OVF_IMPORT_BUILDER_EXECUTABLE, args.docker_image_tag, builder_region)
    return _RunCloudBuild(args, builder, ovf_importer_args, build_tags, output_filter, backoff=backoff, log_location=args.log_location, build_region=builder_region)