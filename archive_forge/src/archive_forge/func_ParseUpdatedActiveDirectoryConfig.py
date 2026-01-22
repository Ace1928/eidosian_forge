from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.netapp import constants
from googlecloudsdk.api_lib.netapp import util as netapp_api_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def ParseUpdatedActiveDirectoryConfig(self, activedirectory_config, domain=None, site=None, dns=None, net_bios_prefix=None, organizational_unit=None, aes_encryption=None, username=None, password=None, backup_operators=None, security_operators=None, kdc_hostname=None, kdc_ip=None, nfs_users_with_ldap=None, ldap_signing=None, encrypt_dc_connections=None, description=None, labels=None):
    """Parses updates into an active directory config."""
    if domain is not None:
        activedirectory_config.domain = domain
    if site is not None:
        activedirectory_config.site = site
    if dns is not None:
        activedirectory_config.dns = dns
    if net_bios_prefix is not None:
        activedirectory_config.netBiosPrefix = net_bios_prefix
    if organizational_unit is not None:
        activedirectory_config.organizationalUnit = organizational_unit
    if aes_encryption is not None:
        activedirectory_config.aesEncryption = aes_encryption
    if username is not None:
        activedirectory_config.username = username
    if password is not None:
        activedirectory_config.password = password
    if backup_operators is not None:
        activedirectory_config.backupOperators = backup_operators
    if security_operators is not None:
        activedirectory_config.securityOperators = security_operators
    if kdc_hostname is not None:
        activedirectory_config.kdcHostname = kdc_hostname
    if kdc_ip is not None:
        activedirectory_config.kdcIp = kdc_ip
    if nfs_users_with_ldap is not None:
        activedirectory_config.nfsUsersWithLdap = nfs_users_with_ldap
    if ldap_signing is not None:
        activedirectory_config.ldapSigning = ldap_signing
    if encrypt_dc_connections is not None:
        activedirectory_config.encryptDcConnections = encrypt_dc_connections
    if description is not None:
        activedirectory_config.description = description
    if labels is not None:
        activedirectory_config.labels = labels
    return activedirectory_config