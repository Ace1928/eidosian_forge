from ansible.plugins.inventory import BaseInventoryPlugin
from ansible.plugins.inventory import Cacheable
from ansible.plugins.inventory import Constructable
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.plugin_utils.base import AWSPluginBase
from ansible_collections.amazon.aws.plugins.plugin_utils.botocore import AnsibleBotocoreError
def update_cached_result(self, path, cache, result):
    if not self.get_option('cache'):
        return
    cache_key = self.get_cache_key(path)
    if cache and cache_key in self._cache:
        return
    self._cache[cache_key] = result