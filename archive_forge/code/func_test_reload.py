import http.client as http
import os
import re
import time
import psutil
import requests
from glance.tests import functional
from glance.tests.utils import execute
def test_reload(self):
    """Test SIGHUP picks up new config values"""

    def check_pids(pre, post=None, workers=2):
        if post is None:
            if len(pre) == workers:
                return True
            else:
                return False
        if len(post) == workers:
            if post.intersection(pre) == set():
                return True
        return False
    self.api_server.fork_socket = False
    self.start_servers(fork_socket=False, **vars(self))
    pre_pids = {}
    post_pids = {}
    path = self._url('/')
    response = requests.get(path)
    self.assertEqual(http.MULTIPLE_CHOICES, response.status_code)
    del response
    pre_pids['api'] = self._get_children('api')
    msg = 'Start timeout'
    for _ in self.ticker(msg):
        pre_pids['api'] = self._get_children('api')
        if check_pids(pre_pids['api'], workers=1):
            break
    set_config_value(self._conffile('api'), 'workers', '2')
    cmd = 'kill -HUP %s' % self._get_parent('api')
    execute(cmd, raise_error=True)
    msg = 'Worker change timeout'
    for _ in self.ticker(msg):
        post_pids['api'] = self._get_children('api')
        if check_pids(pre_pids['api'], post_pids['api']):
            break
    pre_pids['api'] = self._get_children('api')
    set_config_value(self._conffile('api'), 'bind_host', '127.0.0.1')
    cmd = 'kill -HUP %s' % self._get_parent('api')
    execute(cmd, raise_error=True)
    msg = 'http bind_host timeout'
    for _ in self.ticker(msg):
        post_pids['api'] = self._get_children('api')
        if check_pids(pre_pids['api'], post_pids['api']):
            break
    path = self._url('/')
    response = requests.get(path)
    self.assertEqual(http.MULTIPLE_CHOICES, response.status_code)
    del response
    conf_dir = os.path.join(self.test_dir, 'etc')
    log_file = conf_dir + 'new.log'
    self.assertFalse(os.path.exists(log_file))
    set_config_value(self._conffile('api'), 'log_file', log_file)
    cmd = 'kill -HUP %s' % self._get_parent('api')
    execute(cmd, raise_error=True)
    msg = 'No new log file created'
    for _ in self.ticker(msg):
        if os.path.exists(log_file):
            break