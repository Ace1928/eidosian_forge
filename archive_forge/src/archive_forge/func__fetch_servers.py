import collections
import sys
from ansible.errors import AnsibleParserError
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
from ansible_collections.openstack.cloud.plugins.module_utils.openstack import (
def _fetch_servers(self, path, cache):
    cache_key = self._get_cache_prefix(path)
    user_cache_setting = self.get_option('cache')
    attempt_to_read_cache = user_cache_setting and cache
    cache_needs_update = not cache and user_cache_setting
    servers = None
    if attempt_to_read_cache:
        self.display.vvvv('Reading OpenStack inventory cache key {0}'.format(cache_key))
        try:
            servers = self._cache[cache_key]
        except KeyError:
            self.display.vvvv('OpenStack inventory cache not found')
            cache_needs_update = True
    if not attempt_to_read_cache or cache_needs_update:
        self.display.vvvv('Retrieving servers from Openstack clouds')
        clouds_yaml_path = self.get_option('clouds_yaml_path')
        config_files = openstack.config.loader.CONFIG_FILES
        if clouds_yaml_path:
            config_files += clouds_yaml_path
        config = openstack.config.loader.OpenStackConfig(config_files=config_files)
        only_clouds = self.get_option('only_clouds', [])
        if only_clouds:
            if not isinstance(only_clouds, list):
                raise AnsibleParserError('Option only_clouds in OpenStack inventory configuration is not a list')
            cloud_regions = [config.get_one(cloud=cloud) for cloud in only_clouds]
        else:
            cloud_regions = config.get_all()
        clouds = [openstack.connection.Connection(config=cloud_region) for cloud_region in cloud_regions]
        self.display.vvvv('Found {0} OpenStack cloud(s)'.format(len(clouds)))
        self.display.vvvv('Using {0} OpenStack cloud(s)'.format(len(clouds)))
        expand_hostvars = self.get_option('expand_hostvars')
        all_projects = self.get_option('all_projects')
        servers = []

        def _expand_server(server, cloud, volumes):
            server['cloud'] = dict(name=cloud.name)
            region = cloud.config.get_region_name()
            if region:
                server['cloud']['region'] = region
            if not expand_hostvars:
                return server
            server['volumes'] = [v for v in volumes if any((a['server_id'] == server['id'] for a in v['attachments']))]
            return server
        for cloud in clouds:
            if expand_hostvars:
                volumes = [v.to_dict(computed=False) for v in cloud.block_storage.volumes()]
            else:
                volumes = []
            try:
                for server in [_expand_server(server.to_dict(computed=False), cloud, volumes) for server in cloud.compute.servers(all_projects=all_projects, details=True)]:
                    servers.append(server)
            except openstack.exceptions.OpenStackCloudException as e:
                self.display.warning('Fetching servers for cloud {0} failed with: {1}'.format(cloud.name, str(e)))
                if self.get_option('fail_on_errors'):
                    raise
    if cache_needs_update:
        self._cache[cache_key] = servers
    return servers