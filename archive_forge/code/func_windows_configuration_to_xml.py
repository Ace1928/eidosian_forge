import re
import copy
import time
import base64
import random
import collections
from xml.dom import minidom
from datetime import datetime
from xml.sax.saxutils import escape as xml_escape
from libcloud.utils.py3 import ET, httplib, urlparse
from libcloud.utils.py3 import urlquote as url_quote
from libcloud.utils.py3 import _real_unicode, ensure_string
from libcloud.utils.misc import ReprMixin
from libcloud.common.azure import AzureRedirectException, AzureServiceManagementConnection
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.compute.types import NodeState
from libcloud.compute.providers import Provider
@staticmethod
def windows_configuration_to_xml(configuration, xml):
    AzureXmlSerializer.data_to_xml([('ConfigurationSetType', configuration.configuration_set_type)], xml)
    AzureXmlSerializer.data_to_xml([('ComputerName', configuration.computer_name)], xml)
    AzureXmlSerializer.data_to_xml([('AdminPassword', configuration.admin_password)], xml)
    AzureXmlSerializer.data_to_xml([('ResetPasswordOnFirstLogon', configuration.reset_password_on_first_logon, _lower)], xml)
    AzureXmlSerializer.data_to_xml([('EnableAutomaticUpdates', configuration.enable_automatic_updates, _lower)], xml)
    AzureXmlSerializer.data_to_xml([('TimeZone', configuration.time_zone)], xml)
    if configuration.domain_join is not None:
        domain = ET.xml('DomainJoin')
        creds = ET.xml('Credentials')
        domain.appemnd(creds)
        xml.append(domain)
        AzureXmlSerializer.data_to_xml([('Domain', configuration.domain_join.credentials.domain)], creds)
        AzureXmlSerializer.data_to_xml([('Username', configuration.domain_join.credentials.username)], creds)
        AzureXmlSerializer.data_to_xml([('Password', configuration.domain_join.credentials.password)], creds)
        AzureXmlSerializer.data_to_xml([('JoinDomain', configuration.domain_join.join_domain)], domain)
        AzureXmlSerializer.data_to_xml([('MachineObjectOU', configuration.domain_join.machine_object_ou)], domain)
    if configuration.stored_certificate_settings is not None:
        cert_settings = ET.Element('StoredCertificateSettings')
        xml.append(cert_settings)
        for cert in configuration.stored_certificate_settings:
            cert_setting = ET.Element('CertificateSetting')
            cert_settings.append(cert_setting)
            cert_setting.append(AzureXmlSerializer.data_to_xml([('StoreLocation', cert.store_location)]))
            AzureXmlSerializer.data_to_xml([('StoreName', cert.store_name)], cert_setting)
            AzureXmlSerializer.data_to_xml([('Thumbprint', cert.thumbprint)], cert_setting)
    AzureXmlSerializer.data_to_xml([('AdminUsername', configuration.admin_user_name)], xml)
    return xml