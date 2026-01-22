import json
import logging
import os
import re
import subprocess
from googlecloudsdk.third_party.appengine._internal import six_subset
def _GetGitContextTypeFromDomain(url):
    """Returns the context type for the input Git url."""
    if not url:
        return _ContextType.GIT_UNKNOWN
    if not _PROTOCOL_PATTERN.match(url):
        url = 'ssh://' + url
    domain_match = _DOMAIN_PATTERN.match(url)
    protocol = _PROTOCOL_PATTERN.match(url).group('protocol')
    if domain_match:
        domain = domain_match.group('domain')
        if domain == 'google.com':
            return _ContextType.CLOUD_REPO
        elif domain == 'github.com' or domain == 'bitbucket.org':
            if protocol == 'ssh':
                return _ContextType.GIT_KNOWN_HOST_SSH
            else:
                return _ContextType.GIT_KNOWN_HOST
    return _ContextType.GIT_UNKNOWN