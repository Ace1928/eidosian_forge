from __future__ import absolute_import, division, print_function
import time
def wait_for_redis_running(self):
    try:
        response = self._client.redis.get(resource_group_name=self.resource_group, name=self.name)
        status = response.provisioning_state
        polling_times = 0
        while polling_times < self.wait_for_provisioning_polling_times:
            if status.lower() != 'succeeded':
                polling_times += 1
                time.sleep(self.wait_for_provisioning_polling_interval_in_seconds)
                response = self._client.redis.get(resource_group_name=self.resource_group, name=self.name)
                status = response.provisioning_state
            else:
                return True
        self.fail('Azure Cache for Redis is not running after 60 mins.')
    except Exception as e:
        self.fail('Failed to get Azure Cache for Redis: {0}'.format(str(e)))