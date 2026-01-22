from __future__ import absolute_import, division, print_function
import json
import logging
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
def _svc_token_wrap(self, cmd, cmdopts, cmdargs, timeout=10):
    """ Run SVC command with token info added into header
        :param cmd: svc command to run
        :type cmd: string
        :param cmdopts: svc command options, name paramter and value
        :type cmdopts: dict
        :param cmdargs: svc command arguments, non-named paramaters
        :type cmdargs: list
        :param timeout: open_url argument to set timeout for http gateway
        :type timeout: int
        :returns: command results
        """
    if self.token is None:
        self.module.fail_json(msg='No authorize token')
    headers = {'Content-Type': 'application/json', 'X-Auth-Token': self.token}
    return self._svc_rest(method='POST', headers=headers, cmd=cmd, cmdopts=cmdopts, cmdargs=cmdargs, timeout=timeout)