from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import json
import re
from googlecloudsdk.api_lib.container import util as c_util
from googlecloudsdk.command_lib.compute.instance_templates import service_proxy_aux_data
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet import kube_util as hub_kube_util
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def _ModifyInstanceTemplate(args, is_mcp, metadata_args):
    """Modify the instance template to include the service proxy metadata."""
    if metadata_args.asm_labels:
        asm_labels = metadata_args.asm_labels
    else:
        asm_labels = collections.OrderedDict()
    asm_labels[_ISTIO_CANONICAL_SERVICE_NAME_LABEL] = metadata_args.canonical_service
    asm_labels[_ISTIO_CANONICAL_SERVICE_REVISION_LABEL] = metadata_args.canonical_revision
    asm_labels_string = json.dumps(asm_labels, sort_keys=True)
    service_proxy_config = collections.OrderedDict()
    service_proxy_config['mode'] = 'ON'
    service_proxy_config['proxy-spec'] = {'network': metadata_args.network, 'api-server': metadata_args.service_proxy_api_server, 'log-level': 'info'}
    service_proxy_config['service'] = {}
    proxy_config = metadata_args.asm_proxy_config
    if not proxy_config:
        proxy_config = collections.OrderedDict()
    if 'proxyMetadata' not in proxy_config:
        proxy_config['proxyMetadata'] = collections.OrderedDict()
    else:
        proxy_config['proxyMetadata'] = collections.OrderedDict(proxy_config['proxyMetadata'])
    proxy_metadata = proxy_config['proxyMetadata']
    proxy_metadata['ISTIO_META_WORKLOAD_NAME'] = metadata_args.workload_name
    proxy_metadata['POD_NAMESPACE'] = metadata_args.workload_namespace
    proxy_metadata['USE_TOKEN_FOR_CSR'] = 'true'
    proxy_metadata['ISTIO_META_DNS_CAPTURE'] = 'true'
    proxy_metadata['ISTIO_META_AUTO_REGISTER_GROUP'] = metadata_args.workload_name
    proxy_metadata['SERVICE_ACCOUNT'] = metadata_args.service_account
    proxy_metadata['CREDENTIAL_IDENTITY_PROVIDER'] = metadata_args.identity_provider
    if metadata_args.trust_domain:
        proxy_metadata['TRUST_DOMAIN'] = metadata_args.trust_domain
    if metadata_args.mesh_id:
        proxy_metadata['ISTIO_META_MESH_ID'] = metadata_args.mesh_id
    proxy_metadata['ISTIO_META_NETWORK'] = '{}-{}'.format(metadata_args.project_id, metadata_args.network)
    proxy_metadata['CANONICAL_SERVICE'] = metadata_args.canonical_service
    proxy_metadata['CANONICAL_REVISION'] = metadata_args.canonical_revision
    proxy_metadata['ISTIO_METAJSON_LABELS'] = asm_labels_string
    if metadata_args.asm_revision == 'default':
        proxy_metadata['ASM_REVISION'] = ''
    else:
        proxy_metadata['ASM_REVISION'] = metadata_args.asm_revision
    gce_software_declaration = collections.OrderedDict()
    service_proxy_agent_recipe = collections.OrderedDict()
    service_proxy_agent_recipe['name'] = 'install-gce-service-proxy-agent'
    service_proxy_agent_recipe['desired_state'] = 'INSTALLED'
    if is_mcp:
        service_proxy_agent_recipe['installSteps'] = [{'scriptRun': {'script': service_proxy_aux_data.startup_script_for_asm_service_proxy_installer}}]
        proxy_metadata.update(metadata_args.mcp_env_config)
        if _CLOUDRUN_ADDR_KEY in proxy_metadata:
            proxy_metadata[_ISTIO_META_CLOUDRUN_ADDR_KEY] = proxy_metadata[_CLOUDRUN_ADDR_KEY]
        if 'gce-service-proxy-installer-bucket' not in args.metadata:
            args.metadata['gce-service-proxy-installer-bucket'] = _SERVICE_PROXY_INSTALLER_BUCKET_NAME
    else:
        service_proxy_agent_recipe['installSteps'] = [{'scriptRun': {'script': service_proxy_aux_data.startup_script_for_asm_service_proxy.format(ingress_ip=metadata_args.expansionagateway_ip, asm_revision=metadata_args.asm_revision)}}]
        proxy_metadata['ISTIO_META_ISTIO_VERSION'] = metadata_args.asm_version
        args.metadata['rootcert'] = metadata_args.root_cert
        if _GCE_SERVICE_PROXY_AGENT_BUCKET_METADATA not in args.metadata:
            args.metadata[_GCE_SERVICE_PROXY_AGENT_BUCKET_METADATA] = _SERVICE_PROXY_BUCKET_NAME.format(metadata_args.asm_version)
    gce_software_declaration['softwareRecipes'] = [service_proxy_agent_recipe]
    service_proxy_config['asm-config'] = proxy_config
    args.metadata['enable-osconfig'] = 'true'
    args.metadata['enable-guest-attributes'] = 'true'
    args.metadata['osconfig-disabled-features'] = 'tasks'
    args.metadata['gce-software-declaration'] = json.dumps(gce_software_declaration)
    args.metadata['gce-service-proxy'] = json.dumps(service_proxy_config, sort_keys=True)
    if args.labels is None:
        args.labels = collections.OrderedDict()
    args.labels['asm_service_name'] = metadata_args.canonical_service
    args.labels['asm_service_namespace'] = metadata_args.workload_namespace
    if metadata_args.mesh_id:
        args.labels['mesh_id'] = metadata_args.mesh_id
    else:
        project_number = project_util.GetProjectNumber(metadata_args.project_id)
        args.labels['mesh_id'] = 'proj-{}'.format(project_number)
    args.labels['gce-service-proxy'] = 'asm-istiod'