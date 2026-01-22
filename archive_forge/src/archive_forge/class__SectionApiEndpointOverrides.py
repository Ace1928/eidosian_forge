from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import functools
import os
import re
import sys
import textwrap
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.configurations import properties_file as prop_files_lib
from googlecloudsdk.core.docker import constants as const_lib
from googlecloudsdk.core.resource import resource_printer_types as formats
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import http_proxy_types
from googlecloudsdk.core.util import scaled_integer
from googlecloudsdk.generated_clients.apis import apis_map
import six
class _SectionApiEndpointOverrides(_Section):
    """Contains the properties for the 'api-endpoint-overrides' section.

  This overrides what endpoint to use when talking to the given API.
  """

    def __init__(self):
        super(_SectionApiEndpointOverrides, self).__init__('api_endpoint_overrides')
        self.accessapproval = self._Add('accessapproval', command='gcloud access-approval')
        self.accesscontextmanager = self._Add('accesscontextmanager', command='gcloud access-context-manager')
        self.ai = self._Add('ai', command='gcloud ai')
        self.aiplatform = self._Add('aiplatform', command='gcloud ai-platform')
        self.alloydb = self._Add('alloydb', command='gcloud alloydb', hidden=True)
        self.anthosevents = self._Add('anthosevents', command='gcloud anthos')
        self.anthospolicycontrollerstatus_pa = self._Add('anthospolicycontrollerstatus_pa', command='gcloud container fleet policycontroller')
        self.apigateway = self._Add('apigateway', command='gcloud api-gateway')
        self.apigee = self._Add('apigee', command='gcloud apigee')
        self.apigeeregistry = self._Add('apigeeregistry', command='gcloud apigee-registry', hidden=True)
        self.appconfigmanager = self._Add('appconfigmanager', command='gcloud app-config-manager', hidden=True)
        self.appengine = self._Add('appengine', command='gcloud app')
        self.apphub = self._Add('apphub', command='gcloud apphub')
        self.artifactregistry = self._Add('artifactregistry', command='gcloud artifacts')
        self.assuredworkloads = self._Add('assuredworkloads', command='gcloud assured')
        self.auditmanager = self._Add('auditmanager', command='gcloud audit-manager')
        self.authztoolkit = self._Add('authztoolkit', command='gcloud authz-toolkit', hidden=True)
        self.backupdr = self._Add('backupdr', command='gcloud backup-dr', hidden=True)
        self.baremetalsolution = self._Add('baremetalsolution', command='gcloud bms')
        self.batch = self._Add('batch', command='gcloud batch', hidden=True)
        self.beyondcorp = self._Add('beyondcorp', hidden=True)
        self.bigquery = self._Add('bigquery', hidden=True)
        self.bigtableadmin = self._Add('bigtableadmin', command='gcloud bigtable')
        self.binaryauthorization = self._Add('binaryauthorization', command='gcloud container binauthz', hidden=True)
        self.categorymanager = self._Add('categorymanager', hidden=True)
        self.certificatemanager = self._Add('certificatemanager', command='gcloud certificate-manager')
        self.cloudasset = self._Add('cloudasset', command='gcloud asset')
        self.cloudbilling = self._Add('cloudbilling', command='gcloud billing')
        self.cloudbuild = self._Add('cloudbuild', command='gcloud builds')
        self.cloudcommerceconsumerprocurement = self._Add('cloudcommerceconsumerprocurement', command='gcloud commerce-procurement')
        self.clouddebugger = self._Add('clouddebugger', command='gcloud debug')
        self.clouddeploy = self._Add('clouddeploy', command='gcloud deploy')
        self.clouderrorreporting = self._Add('clouderrorreporting', command='gcloud error-reporting')
        self.cloudfunctions = self._Add('cloudfunctions', command='gcloud functions')
        self.cloudidentity = self._Add('cloudidentity', command='gcloud identity')
        self.cloudiot = self._Add('cloudiot', command='gcloud iot')
        self.cloudkms = self._Add('cloudkms', command='gcloud kms')
        self.cloudnumberregistry = self._Add('cloudnumberregistry', command='gcloud cloudnumberregistry', hidden=True)
        self.cloudquotas = self._Add('cloudquotas', command='gcloud quotas', hidden=True)
        self.cloudresourcemanager = self._Add('cloudresourcemanager', command='gcloud projects')
        self.cloudresourcesearch = self._Add('cloudresourcesearch', hidden=True)
        self.cloudscheduler = self._Add('cloudscheduler', command='gcloud scheduler')
        self.cloudtasks = self._Add('cloudtasks', command='gcloud tasks')
        self.cloudtrace = self._Add('cloudtrace', command='gcloud trace')
        self.composer = self._Add('composer', command='gcloud composer')
        self.compute = self._Add('compute', help_text='Overrides API endpoint for `gcloud compute` command group. For Private Service Connect usage, see https://cloud.google.com/vpc/docs/configure-private-service-connect-apis#using-endpoints')
        self.configdelivery = self._Add('configdelivery', command='gcloud container fleet packages', hidden=True)
        self.connectgateway = self._Add('connectgateway', hidden=True)
        self.container = self._Add('container', command='gcloud container')
        self.containeranalysis = self._Add('containeranalysis', hidden=True)
        self.datacatalog = self._Add('datacatalog', command='gcloud data-catalog')
        self.dataflow = self._Add('dataflow', command='gcloud dataflow')
        self.datafusion = self._Add('datafusion', command='gcloud data-fusion')
        self.datamigration = self._Add('datamigration', command='gcloud database-migration')
        self.datapol = self._Add('datapol', hidden=True)
        self.datapipelines = self._Add('datapipelines', command='gcloud datapipelines')
        self.dataplex = self._Add('dataplex', command='gcloud dataplex')
        self.dataproc = self._Add('dataproc', command='gcloud dataproc')
        self.dataprocgdc = self._Add('dataprocgdc', hidden=True)
        self.datastore = self._Add('datastore', command='gcloud datastore')
        self.datastream = self._Add('datastream', command='gcloud datastream')
        self.deploymentmanager = self._Add('deploymentmanager', command='gcloud deployment-manager')
        self.discovery = self._Add('discovery', hidden=True)
        self.dns = self._Add('dns', command='gcloud dns')
        self.domains = self._Add('domains', command='gcloud domains')
        self.edgecontainer = self._Add('edgecontainer', command='gcloud edge-container')
        self.edgenetwork = self._Add('edgenetwork', command='gcloud edge-cloud networking', hidden=True)
        self.eventarc = self._Add('eventarc', command='gcloud eventarc')
        self.eventarcpublishing = self._Add('eventarcpublishing', command='gcloud eventarc publish')
        self.faultinjectiontesting = self._Add('faultinjectiontesting', command='gcloud fault-injection')
        self.file = self._Add('file', command='gcloud filestore')
        self.firestore = self._Add('firestore', command='gcloud firestore')
        self.genomics = self._Add('genomics', command='gcloud genomics')
        self.gkebackup = self._Add('gkebackup', hidden=True)
        self.gkehub = self._Add('gkehub', hidden=True)
        self.gkemulticloud = self._Add('gkemulticloud', help_text='Overrides API endpoint for `gcloud container aws`, `gcloud container azure` and `gcloud container attached` command groups.')
        self.gkeonprem = self._Add('gkeonprem', hidden=True)
        self.healthcare = self._Add('healthcare', command='gcloud healthcare')
        self.iam = self._Add('iam', command='gcloud iam')
        self.iamcredentials = self._Add('iamcredentials', command='gcloud iam')
        self.iap = self._Add('iap', command='gcloud iap')
        self.ids = self._Add('ids', command='gcloud ids')
        self.krmapihosting = self._Add('krmapihosting', command='gcloud anthos config controller')
        self.kubernetespolicy = self._Add('kubernetespolicy', hidden=True)
        self.inframanager = self._Add('config', command='gcloud infra-manager')
        self.language = self._Add('language', command='gcloud ml language')
        self.lifesciences = self._Add('lifesciences', command='gcloud lifesciences')
        self.logging = self._Add('logging', command='gcloud logging')
        self.looker = self._Add('looker', command='gcloud looker')
        self.managedidentities = self._Add('managedidentities', command='gcloud active-directory')
        self.manager = self._Add('manager', hidden=True)
        self.marketplacesolutions = self._Add('marketplacesolutions', command='gcloud mps')
        self.mediaasset = self._Add('mediaasset', command='gcloud media')
        self.memcache = self._Add('memcache', command='gcloud memcache')
        self.messagestreams = self._Add('messagestreams', command='gcloud messagestreams', hidden=True)
        self.metastore = self._Add('metastore', command='gcloud metastore')
        self.ml = self._Add('ml', hidden=True)
        self.monitoring = self._Add('monitoring', command='gcloud monitoring')
        self.netapp = self._Add('netapp', command='gcloud netapp')
        self.networkconnectivity = self._Add('networkconnectivity', command='gcloud network-connectivity')
        self.networkmanagement = self._Add('networkmanagement', command='gcloud network-management')
        self.networksecurity = self._Add('networksecurity', command='gcloud network-security')
        self.networkservices = self._Add('networkservices', command='gcloud network-services')
        self.notebooks = self._Add('notebooks', command='gcloud notebooks')
        self.ondemandscanning = self._Add('ondemandscanning', hidden=True)
        self.orglifecycle = self._Add('orglifecycle', command='gcloud orglifecycle', hidden=True)
        self.orgpolicy = self._Add('orgpolicy', command='gcloud org-policies')
        self.osconfig = self._Add('osconfig', hidden=True)
        self.oslogin = self._Add('oslogin', hidden=True)
        self.parallelstore = self._Add('parallelstore', hidden=True)
        self.policyanalyzer = self._Add('policyanalyzer', command='policy-intelligence')
        self.policysimulator = self._Add('policysimulator', hidden=True)
        self.policytroubleshooter = self._Add('policytroubleshooter', hidden=True)
        self.privateca = self._Add('privateca', command='gcloud privateca')
        self.privilegedaccessmanager = self._Add('pam', command='gcloud pam')
        self.publicca = self._Add('publicca', command='gcloud publicca')
        self.pubsub = self._Add('pubsub', command='gcloud pubsub')
        self.pubsublite = self._Add('pubsublite', hidden=True)
        self.recaptcha = self._Add('recaptchaenterprise', command='gcloud recaptcha')
        self.recommender = self._Add('recommender', command='gcloud recommender')
        self.redis = self._Add('redis', command='gcloud redis')
        self.remotebuildexecution = self._Add('remotebuildexecution', hidden=True)
        self.replicapoolupdater = self._Add('replicapoolupdater', hidden=True)
        self.resourcesettings = self._Add('resourcesettings', command='gcloud resource-settings')
        self.run = self._Add('run', command='gcloud run')
        self.runapps = self._Add('runapps', hidden=True)
        self.runtimeconfig = self._Add('runtimeconfig', command='gcloud runtime-config')
        self.sasportal = self._Add('sasportal', hidden=True)
        self.scc = self._Add('securitycenter', command='gcloud scc')
        self.sddc = self._Add('sddc', command='gcloud vmware sddc')
        self.seclm = self._Add('seclm', command='gcloud seclm', hidden=True)
        self.secrets = self._Add('secretmanager', command='gcloud secrets')
        self.securedlandingzone = self._Add('securedlandingzone', hidden=True, command='gcloud scc slz-overwatch')
        self.securesourcemanager = self._Add('securesourcemanager', hidden=True)
        self.securitycentermanagement = self._Add('securitycentermanagement', command='gcloud scc manage', hidden=True)
        self.securityposture = self._Add('securityposture', hidden=True)
        self.servicedirectory = self._Add('servicedirectory', command='gcloud service-directory')
        self.servicemanagement = self._Add('servicemanagement', command='gcloud endpoints')
        self.serviceregistry = self._Add('serviceregistry', hidden=True)
        self.serviceusage = self._Add('serviceusage', hidden=True)
        self.source = self._Add('source', hidden=True)
        self.sourcerepo = self._Add('sourcerepo', command='gcloud source')
        self.spanner = self._Add('spanner', help_text='Overrides API endpoint for `gcloud spanner` command group. For spanner emulator usage, see https://cloud.google.com/spanner/docs/emulator#using_the_gcloud_cli_with_the_emulator')
        self.speech = self._Add('speech', command='gcloud ml speech')
        self.sql = self._Add('sql', command='gcloud sql')
        self.storage = self._Add('storage', command='gcloud storage')
        self.storageinsights = self._Add('storageinsights', command='gcloud storage insights', hidden=True)
        self.stream = self._Add('stream', hidden=True)
        self.telcoautomation = self._Add('telcoautomation', hidden=True)
        self.testing = self._Add('testing', command='gcloud firebase test')
        self.toolresults = self._Add('toolresults', hidden=True)
        self.tpu = self._Add('tpu', hidden=True)
        self.transfer = self._Add('transfer', command='gcloud transfer')
        self.vision = self._Add('vision', command='gcloud ml vision')
        self.vmmigration = self._Add('vmmigration', command='gcloud migration vms')
        self.vmwareengine = self._Add('vmwareengine', command='gcloud vmware')
        self.vpcaccess = self._Add('vpcaccess', hidden=True)
        self.workflowexecutions = self._Add('workflowexecutions', command='gcloud workflows executions')
        self.workflows = self._Add('workflows', command='gcloud workflows')
        self.workstations = self._Add('workstations', command='gcloud workstations')

    def EndpointValidator(self, value):
        """Checks to see if the endpoint override string is valid."""
        if value is None:
            return
        if not _VALID_ENDPOINT_OVERRIDE_REGEX.match(value):
            raise InvalidValueError("The endpoint_overrides property must be an absolute URI beginning with http:// or https:// and ending with a trailing '/'. [{value}] is not a valid endpoint override.".format(value=value))

    def _Add(self, name, help_text=None, hidden=False, command=None):
        if not help_text and command:
            help_text = 'Overrides API endpoint for `{}` command group.'.format(command)
        default_endpoint = self.GetDefaultEndpoint(name)
        if command and default_endpoint:
            help_text = f'{help_text} Defaults to `{default_endpoint}`'
        return super(_SectionApiEndpointOverrides, self)._Add(name, help_text=help_text, hidden=hidden, validator=self.EndpointValidator)

    def GetDefaultEndpoint(self, api_name):
        """Returns the BASE_URL for the respective api and version."""
        api = apis_map.MAP.get(api_name)
        if api:
            for api_version in api:
                api_def = api.get(api_version)
                if api_def.default_version and api_def.apitools:
                    return api_def.apitools.base_url