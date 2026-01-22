from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions as gcloud_exceptions
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.scc.settings import exceptions as scc_exceptions
from googlecloudsdk.core import properties
def DescribeServiceExplicit(self, args):
    """Describe effective service settings of organization/folder/project."""
    FallBackFlags(args)
    path = GenerateParent(args) + SERVICES_ENDPOINTS[args.service]
    try:
        if args.organization:
            if args.service == 'web-security-scanner':
                request_message = self.message_module.SecuritycenterOrganizationsGetWebSecurityScannerSettingsRequest(name=path)
                return self.service_client.organizations.GetWebSecurityScannerSettings(request_message)
            elif args.service == 'security-health-analytics':
                request_message = self.message_module.SecuritycenterOrganizationsGetSecurityHealthAnalyticsSettingsRequest(name=path)
                return self.service_client.organizations.GetSecurityHealthAnalyticsSettings(request_message)
            elif args.service == 'container-threat-detection':
                request_message = self.message_module.SecuritycenterOrganizationsGetContainerThreatDetectionSettingsRequest(name=path)
                return self.service_client.organizations.GetContainerThreatDetectionSettings(request_message)
            elif args.service == 'event-threat-detection':
                request_message = self.message_module.SecuritycenterOrganizationsGetEventThreatDetectionSettingsRequest(name=path)
                return self.service_client.organizations.GetEventThreatDetectionSettings(request_message)
            elif args.service == 'virtual-machine-threat-detection':
                request_message = self.message_module.SecuritycenterOrganizationsGetVirtualMachineThreatDetectionSettingsRequest(name=path)
                return self.service_client.organizations.GetVirtualMachineThreatDetectionSettings(request_message)
            elif args.service == 'rapid-vulnerability-detection':
                request_message = self.message_module.SecuritycenterOrganizationsGetRapidVulnerabilityDetectionSettingsRequest(name=path)
                return self.service_client.organizations.GetRapidVulnerabilityDetectionSettings(request_message)
        elif args.project:
            if args.service == 'web-security-scanner':
                request_message = self.message_module.SecuritycenterProjectsGetWebSecurityScannerSettingsRequest(name=path)
                return self.service_client.projects.GetWebSecurityScannerSettings(request_message)
            elif args.service == 'security-health-analytics':
                request_message = self.message_module.SecuritycenterProjectsGetSecurityHealthAnalyticsSettingsRequest(name=path)
                return self.service_client.projects.GetSecurityHealthAnalyticsSettings(request_message)
            elif args.service == 'container-threat-detection':
                request_message = self.message_module.SecuritycenterProjectsGetContainerThreatDetectionSettingsRequest(name=path)
                return self.service_client.projects.GetContainerThreatDetectionSettings(request_message)
            elif args.service == 'event-threat-detection':
                request_message = self.message_module.SecuritycenterProjectsGetEventThreatDetectionSettingsRequest(name=path)
                return self.service_client.projects.GetEventThreatDetectionSettings(request_message)
            elif args.service == 'virtual-machine-threat-detection':
                request_message = self.message_module.SecuritycenterProjectsGetVirtualMachineThreatDetectionSettingsRequest(name=path)
                return self.service_client.projects.GetVirtualMachineThreatDetectionSettings(request_message)
            elif args.service == 'rapid-vulnerability-detection':
                request_message = self.message_module.SecuritycenterProjectsGetRapidVulnerabilityDetectionSettingsRequest(name=path)
                return self.service_client.projects.GetRapidVulnerabilityDetectionSettings(request_message)
        elif args.folder:
            if args.service == 'web-security-scanner':
                request_message = self.message_module.SecuritycenterFoldersGetWebSecurityScannerSettingsRequest(name=path)
                return self.service_client.folders.GetWebSecurityScannerSettings(request_message)
            elif args.service == 'security-health-analytics':
                request_message = self.message_module.SecuritycenterFoldersGetSecurityHealthAnalyticsSettingsRequest(name=path)
                return self.service_client.folders.GetSecurityHealthAnalyticsSettings(request_message)
            elif args.service == 'container-threat-detection':
                request_message = self.message_module.SecuritycenterFoldersGetContainerThreatDetectionSettingsRequest(name=path)
                return self.service_client.folders.GetContainerThreatDetectionSettings(request_message)
            elif args.service == 'event-threat-detection':
                request_message = self.message_module.SecuritycenterFoldersGetEventThreatDetectionSettingsRequest(name=path)
                return self.service_client.folders.GetEventThreatDetectionSettings(request_message)
            elif args.service == 'virtual-machine-threat-detection':
                request_message = self.message_module.SecuritycenterFoldersGetVirtualMachineThreatDetectionSettingsRequest(name=path)
                return self.service_client.folders.GetVirtualMachineThreatDetectionSettings(request_message)
            elif args.service == 'rapid-vulnerability-detection':
                request_message = self.message_module.SecuritycenterFoldersGetRapidVulnerabilityDetectionSettingsRequest(name=path)
                return self.service_client.folders.GetRapidVulnerabilityDetectionSettings(request_message)
    except exceptions.HttpError as err:
        gcloud_exceptions.core_exceptions.reraise(gcloud_exceptions.HttpException(err, error_format='Status code [{status_code}]. {message}.'))