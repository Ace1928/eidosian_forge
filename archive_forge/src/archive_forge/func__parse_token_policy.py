import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
def _parse_token_policy(self, token):
    """validate policy token"""
    if len(token) <= 1 or token[-1:] == token[0]:
        self.dist_fatal("'$' must stuck in the begin of policy name")
    token = token[1:]
    if token not in self._parse_policies:
        self.dist_fatal("'%s' is an invalid policy name, available policies are" % token, self._parse_policies.keys())
    return token