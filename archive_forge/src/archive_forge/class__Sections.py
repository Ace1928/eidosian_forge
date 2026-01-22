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
class _Sections(object):
    """Represents the available sections in the properties file.

  Attributes:
    access_context_manager: Section, The section containing access context
      manager properties for the Cloud SDK.
    accessibility: Section, The section containing accessibility properties for
      the Cloud SDK.
    ai: Section, The section containing ai properties for the Cloud SDK.
    ai_platform: Section, The section containing ai platform properties for the
      Cloud SDK.
    api_client_overrides: Section, The section containing API client override
      properties for the Cloud SDK.
    api_endpoint_overrides: Section, The section containing API endpoint
      override properties for the Cloud SDK.
    app: Section, The section containing app properties for the Cloud SDK.
    auth: Section, The section containing auth properties for the Cloud SDK.
    batch: Section, The section containing batch properties for the Cloud SDK.
    billing: Section, The section containing billing properties for the Cloud
      SDK.
    builds: Section, The section containing builds properties for the Cloud SDK.
    artifacts: Section, The section containing artifacts properties for the
      Cloud SDK.
    code: Section, The section containing local development properties for Cloud
      SDK.
    component_manager: Section, The section containing properties for the
      component_manager.
    composer: Section, The section containing composer properties for the Cloud
      SDK.
    compute: Section, The section containing compute properties for the Cloud
      SDK.
    config_delivery: Section, The section containing properties for Config
      Delivery.
    container: Section, The section containing container properties for the
      Cloud SDK.
    container_attached: Section, The section containing properties for Attached
      clusters.
    container_aws: Section, The section containing properties for Anthos
      clusters on AWS.
    container_azure: Section, The section containing properties for Anthos
      clusters on Azure.
    container_bare_metal: Section, The section containing properties for Anthos
      clusters on Bare Metal.
    container_vmware: Section, The section containing properties for Anthos
      clusters on VMware.
    context_aware: Section, The section containing context aware access
      configurations for the Cloud SDK.
    core: Section, The section containing core properties for the Cloud SDK.
    ssh: Section, The section containing ssh-related properties.
    scc: Section, The section containing scc properties for the Cloud SDK.
    deploy: Secion, The secion containing cloud deploy related properties for
      the Cloud SDK.
    dataproc: Section, The section containing dataproc properties for the Cloud
      SDK.
    dataflow: Section, The section containing dataflow properties for the Cloud
      SDK.
    datafusion: Section, The section containing datafusion properties for the
      Cloud SDK.
    datapipelines: Section, The section containing datapipelines properties for
      the cloud SDK.
    dataplex: Section, The section containing dataplex properties for the Cloud
      SDK.
    declarative: Section, The section containing properties for declarative
      workflows in the Cloud SDK.
    default_section: Section, The main section of the properties file (core).
    deployment_manager: Section, The section containing deployment_manager
      properties for the Cloud SDK.
    devshell: Section, The section containing devshell properties for the Cloud
      SDK.
    diagnostics: Section, The section containing diagnostics properties for the
      Cloud SDK.
    edge_container: Section, The section containing edgecontainer properties for
      the Cloud SDK.
    emulator: Section, The section containing emulator properties for the Cloud
      SDK.
    eventarc: Section, The section containing eventarc properties for the Cloud
      SDK.
    experimental: Section, The section containing experimental properties for
      the Cloud SDK.
    filestore: Section, The section containing filestore properties for the
      Cloud SDK.
    functions: Section, The section containing functions properties for the
      Cloud SDK.
    gcloudignore: Section, The section containing gcloudignore properties for
      the Cloud SDK.
    gkebackup: Section, The section containing gkebackup properties for the
      Cloud SDK.
    gkehub: Section, The section containing gkehub properties for the Cloud SDK.
    healthcare: Section, The section containing healthcare properties for the
      Cloud SDK.
    inframanager: Section, The section containing Infra Manager properties for
      the Cloud SDK.
    interactive: Section, The section containing interactive properties for the
      Cloud SDK.
    kuberun: Section, The section containing kuberun properties for the Cloud
      SDK.
    lifesciences: Section, The section containing lifesciencs properties for the
      Cloud SDK.
    looker: Section, The section containing looker properties for the Cloud SDK.
    media_asset: Section, the section containing mediaasset protperties for the
      Cloud SDK.
    memcache: Section, The section containing memcache properties for the Cloud
      SDK.
    metastore: Section, The section containing metastore properties for the
      Cloud SDK.
    metrics: Section, The section containing metrics properties for the Cloud
      SDK.
    ml_engine: Section, The section containing ml_engine properties for the
      Cloud SDK.
    mps: Section, The section containing mps properties for the Cloud SDK.
    netapp: Section, The section containing netapp properties for the Cloud SDK.
    notebooks: Section, The section containing notebook properties for the Cloud
      SDK.
    privateca: Section, The section containing privateca properties for the
      Cloud SDK.
    proxy: Section, The section containing proxy properties for the Cloud SDK.
    pubsub: Section, The section containing pubsub properties for the Cloud SDK.
    recaptcha: Section, The section containing recaptcha properties for the
      Cloud SDK.
    redis: Section, The section containing redis properties for the Cloud SDK.
    resource_policy: Section, The section containing resource policy
      configurations for the Cloud SDK.
    run: Section, The section containing run properties for the Cloud SDK.
    runapps: Section, The section containing runapps properties for the Cloud
      SDK.
    secrets: Section, The section containing secretmanager properties for the
      Cloud SDK.
    spanner: Section, The section containing spanner properties for the Cloud
      SDK.
    storage: Section, The section containing storage properties for the Cloud
      SDK.
    survey: Section, The section containing survey properties for the Cloud SDK.
    test: Section, The section containing test properties for the Cloud SDK.
    transfer: Section, The section containing transfer properties for the Cloud
      SDK.
    transport: Section, The section containing transport properties for the
      Cloud SDK.
    transcoder: Section, The section containing transcoder properties for the
      Cloud SDK.
    vmware: Section, The section containing vmware properties for the Cloud SDK.
    web3: Section, the section containing web3 properties for the Cloud SDK.
    workflows: Section, The section containing workflows properties for the
      Cloud SDK.
    workstations: Section, The section containing workstations properties for
      the Cloud SDK.
  """

    class _ValueFlag(object):

        def __init__(self, value, flag):
            self.value = value
            self.flag = flag

    def __init__(self):
        self.access_context_manager = _SectionAccessContextManager()
        self.accessibility = _SectionAccessibility()
        self.ai = _SectionAi()
        self.ai_platform = _SectionAiPlatform()
        self.api_client_overrides = _SectionApiClientOverrides()
        self.api_endpoint_overrides = _SectionApiEndpointOverrides()
        self.app = _SectionApp()
        self.artifacts = _SectionArtifacts()
        self.auth = _SectionAuth()
        self.batch = _SectionBatch()
        self.billing = _SectionBilling()
        self.builds = _SectionBuilds()
        self.code = _SectionCode()
        self.component_manager = _SectionComponentManager()
        self.composer = _SectionComposer()
        self.compute = _SectionCompute()
        self.config_delivery = _SectionConfigDelivery()
        self.container = _SectionContainer()
        self.container_attached = _SectionContainerAttached()
        self.container_aws = _SectionContainerAws()
        self.container_azure = _SectionContainerAzure()
        self.container_vmware = _SectionContainerVmware()
        self.container_bare_metal = _SectionContainerBareMetal()
        self.context_aware = _SectionContextAware()
        self.core = _SectionCore()
        self.ssh = _SectionSsh()
        self.scc = _SectionScc()
        self.deploy = _SectionDeploy()
        self.dataproc = _SectionDataproc()
        self.dataflow = _SectionDataflow()
        self.datafusion = _SectionDatafusion()
        self.datapipelines = _SectionDataPipelines()
        self.dataplex = _SectionDataplex()
        self.declarative = _SectionDeclarative()
        self.deployment_manager = _SectionDeploymentManager()
        self.devshell = _SectionDevshell()
        self.diagnostics = _SectionDiagnostics()
        self.edge_container = _SectionEdgeContainer()
        self.emulator = _SectionEmulator()
        self.eventarc = _SectionEventarc()
        self.experimental = _SectionExperimental()
        self.filestore = _SectionFilestore()
        self.functions = _SectionFunctions()
        self.gcloudignore = _SectionGcloudignore()
        self.gkehub = _SectionGkeHub()
        self.gkebackup = _SectionGkebackup()
        self.healthcare = _SectionHealthcare()
        self.inframanager = _SectionInfraManager()
        self.interactive = _SectionInteractive()
        self.kuberun = _SectionKubeRun()
        self.lifesciences = _SectionLifeSciences()
        self.looker = _SectionLooker()
        self.media_asset = _SectionMediaAsset()
        self.memcache = _SectionMemcache()
        self.metastore = _SectionMetastore()
        self.metrics = _SectionMetrics()
        self.ml_engine = _SectionMlEngine()
        self.mps = _SectionMps()
        self.netapp = _SectionNetapp()
        self.notebooks = _SectionNotebooks()
        self.privateca = _SectionPrivateCa()
        self.proxy = _SectionProxy()
        self.pubsub = _SectionPubsub()
        self.recaptcha = _SectionRecaptcha()
        self.redis = _SectionRedis()
        self.resource_policy = _SectionResourcePolicy()
        self.run = _SectionRun()
        self.runapps = _SectionRunApps()
        self.secrets = _SectionSecrets()
        self.spanner = _SectionSpanner()
        self.storage = _SectionStorage()
        self.survey = _SectionSurvey()
        self.test = _SectionTest()
        self.transfer = _SectionTransfer()
        self.transport = _SectionTransport()
        self.transcoder = _SectionTranscoder()
        self.vmware = _SectionVmware()
        self.web3 = _SectionWeb3()
        self.workflows = _SectionWorkflows()
        self.workstations = _SectionWorkstations()
        sections = [self.access_context_manager, self.accessibility, self.ai, self.ai_platform, self.api_client_overrides, self.api_endpoint_overrides, self.app, self.auth, self.batch, self.billing, self.builds, self.artifacts, self.code, self.component_manager, self.composer, self.compute, self.config_delivery, self.container, self.container_attached, self.container_aws, self.container_azure, self.container_bare_metal, self.container_vmware, self.context_aware, self.core, self.ssh, self.scc, self.dataproc, self.dataflow, self.datafusion, self.datapipelines, self.dataplex, self.deploy, self.deployment_manager, self.devshell, self.diagnostics, self.edge_container, self.emulator, self.eventarc, self.experimental, self.filestore, self.functions, self.gcloudignore, self.gkebackup, self.healthcare, self.inframanager, self.interactive, self.kuberun, self.lifesciences, self.looker, self.media_asset, self.memcache, self.metastore, self.metrics, self.ml_engine, self.mps, self.netapp, self.notebooks, self.pubsub, self.privateca, self.proxy, self.recaptcha, self.redis, self.resource_policy, self.run, self.runapps, self.secrets, self.spanner, self.storage, self.survey, self.test, self.transport, self.transcoder, self.vmware, self.web3, self.workflows, self.workstations]
        self.__sections = {section.name: section for section in sections}
        self.__invocation_value_stack = [{}]

    @property
    def default_section(self):
        return self.core

    def __iter__(self):
        return iter(self.__sections.values())

    def PushInvocationValues(self):
        self.__invocation_value_stack.append({})

    def PopInvocationValues(self):
        self.__invocation_value_stack.pop()

    def SetInvocationValue(self, prop, value, flag):
        """Set the value of this property for this command, using a flag.

    Args:
      prop: _Property, The property with an explicit value.
      value: str, The value that should be returned while this command is
        running.
      flag: str, The flag that a user can use to set the property, reported if
        it was required at some point but not set by the command line.
    """
        value_flags = self.GetLatestInvocationValues()
        if value:
            prop.Validate(value)
        value_flags[prop] = _Sections._ValueFlag(value, flag)

    def GetLatestInvocationValues(self):
        return self.__invocation_value_stack[-1]

    def GetInvocationStack(self):
        return self.__invocation_value_stack

    def Section(self, section):
        """Gets a section given its name.

    Args:
      section: str, The section for the desired property.

    Returns:
      Section, The section corresponding to the given name.

    Raises:
      NoSuchPropertyError: If the section is not known.
    """
        try:
            return self.__sections[section]
        except KeyError:
            raise NoSuchPropertyError('Section "{section}" does not exist.'.format(section=section))

    def AllSections(self, include_hidden=False):
        """Gets a list of all registered section names.

    Args:
      include_hidden: bool, True to include hidden properties in the result.

    Returns:
      [str], The section names.
    """
        return [name for name, value in six.iteritems(self.__sections) if not value.is_hidden or include_hidden]

    def AllValues(self, list_unset=False, include_hidden=False, properties_file=None, only_file_contents=False):
        """Gets the entire collection of property values for all sections.

    Args:
      list_unset: bool, If True, include unset properties in the result.
      include_hidden: bool, True to include hidden properties in the result. If
        a property has a value set but is hidden, it will be included regardless
        of this setting.
      properties_file: PropertyFile, the file to read settings from.  If None
        the active property file will be used.
      only_file_contents: bool, True if values should be taken only from the
        properties file, false if flags, env vars, etc. should be consulted too.
        Mostly useful for listing file contents.

    Returns:
      {str:{str:str}}, A dict of sections to dicts of properties to values.
    """
        result = {}
        for section in self:
            section_result = section.AllValues(list_unset=list_unset, include_hidden=include_hidden, properties_file=properties_file, only_file_contents=only_file_contents)
            if section_result:
                result[section.name] = section_result
        return result

    def AllPropertyValues(self, list_unset=False, include_hidden=False, properties_file=None, only_file_contents=False):
        """Gets the entire collection of property values for all sections.

    Args:
      list_unset: bool, If True, include unset properties in the result.
      include_hidden: bool, True to include hidden properties in the result. If
        a property has a value set but is hidden, it will be included regardless
        of this setting.
      properties_file: PropertyFile, the file to read settings from.  If None
        the active property file will be used.
      only_file_contents: bool, True if values should be taken only from the
        properties file, false if flags, env vars, etc. should be consulted too.
        Mostly useful for listing file contents.

    Returns:
      {str:{str:PropertyValue}}, A dict of sections to dicts of properties to
        property values.
    """
        result = {}
        for section in self:
            section_result = section.AllPropertyValues(list_unset=list_unset, include_hidden=include_hidden, properties_file=properties_file, only_file_contents=only_file_contents)
            if section_result:
                result[section.name] = section_result
        return result

    def GetHelpString(self):
        """Gets a string with the help contents for all properties and descriptions.

    Returns:
      str, The string for the man page section.
    """
        messages = []
        sections = [self.default_section]
        default_section_name = self.default_section.name
        sections.extend(sorted([s for name, s in six.iteritems(self.__sections) if name != default_section_name and (not s.is_hidden)]))
        for section in sections:
            props = sorted([p for p in section if not p.is_hidden])
            if not props:
                continue
            messages.append('_{section}_::'.format(section=section.name))
            for prop in props:
                messages.append('*{prop}*:::\n\n{text}'.format(prop=prop.name, text=prop.help_text))
        return '\n\n\n'.join(messages)