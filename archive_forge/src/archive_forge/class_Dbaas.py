import httplib2
import logging
import os
import sys
import time
from troveclient.compat import auth
from troveclient.compat import exceptions
class Dbaas(object):
    """Top-level object to access the Rackspace Database as a Service API.

    Create an instance with your creds::

        >> red = Dbaas(USERNAME, API_KEY, TENANT, AUTH_URL, SERVICE_NAME,                         SERVICE_URL)

    Then call methods on its managers::

        >> red.instances.list()
        ...
        >> red.flavors.list()
        ...

    &c.
    """

    def __init__(self, username, api_key, tenant=None, auth_url=None, service_type='database', service_name=None, service_url=None, insecure=False, auth_strategy='keystone', region_name=None, client_cls=TroveHTTPClient):
        from troveclient.compat import versions
        from troveclient.v1 import accounts
        from troveclient.v1 import backups
        from troveclient.v1 import clusters
        from troveclient.v1 import configurations
        from troveclient.v1 import databases
        from troveclient.v1 import datastores
        from troveclient.v1 import diagnostics
        from troveclient.v1 import flavors
        from troveclient.v1 import hosts
        from troveclient.v1 import instances
        from troveclient.v1 import limits
        from troveclient.v1 import management
        from troveclient.v1 import metadata
        from troveclient.v1 import modules
        from troveclient.v1 import quota
        from troveclient.v1 import root
        from troveclient.v1 import security_groups
        from troveclient.v1 import storage
        from troveclient.v1 import users
        self.client = client_cls(username, api_key, tenant, auth_url, service_type=service_type, service_name=service_name, service_url=service_url, insecure=insecure, auth_strategy=auth_strategy, region_name=region_name)
        self.versions = versions.Versions(self)
        self.databases = databases.Databases(self)
        self.flavors = flavors.Flavors(self)
        self.instances = instances.Instances(self)
        self.limits = limits.Limits(self)
        self.users = users.Users(self)
        self.root = root.Root(self)
        self.hosts = hosts.Hosts(self)
        self.quota = quota.Quotas(self)
        self.backups = backups.Backups(self)
        self.clusters = clusters.Clusters(self)
        self.security_groups = security_groups.SecurityGroups(self)
        self.security_group_rules = security_groups.SecurityGroupRules(self)
        self.datastores = datastores.Datastores(self)
        self.datastore_versions = datastores.DatastoreVersions(self)
        self.datastore_version_members = datastores.DatastoreVersionMembers(self)
        self.storage = storage.StorageInfo(self)
        self.management = management.Management(self)
        self.mgmt_cluster = management.MgmtClusters(self)
        self.mgmt_flavor = management.MgmtFlavors(self)
        self.accounts = accounts.Accounts(self)
        self.diagnostics = diagnostics.DiagnosticsInterrogator(self)
        self.hwinfo = diagnostics.HwInfoInterrogator(self)
        self.configurations = configurations.Configurations(self)
        config_parameters = configurations.ConfigurationParameters(self)
        self.configuration_parameters = config_parameters
        self.metadata = metadata.Metadata(self)
        self.modules = modules.Modules(self)
        self.mgmt_configs = management.MgmtConfigurationParameters(self)
        self.mgmt_datastore_versions = management.MgmtDatastoreVersions(self)

        class Mgmt(object):

            def __init__(self, dbaas):
                self.instances = dbaas.management
                self.hosts = dbaas.hosts
                self.accounts = dbaas.accounts
                self.storage = dbaas.storage
                self.datastore_version = dbaas.mgmt_datastore_versions
        self.mgmt = Mgmt(self)

    def set_management_url(self, url):
        self.client.management_url = url

    def get_timings(self):
        return self.client.get_timings()

    def authenticate(self):
        """Authenticate against the server.

        This is called to perform an authentication to retrieve a token.

        Returns on success; raises :exc:`exceptions.Unauthorized` if the
        credentials are wrong.
        """
        self.client.authenticate()