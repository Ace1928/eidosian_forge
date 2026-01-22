from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import http_proxy
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import http_proxy_types
import httplib2
def _ProxySetupWalkthrough():
    """Walks user through setting up gcloud proxy properties."""
    proxy_type_options = sorted((t.upper() for t in http_proxy_types.PROXY_TYPE_MAP))
    proxy_type_idx = console_io.PromptChoice(proxy_type_options, message='Select the proxy type:')
    if proxy_type_idx is None:
        return False
    proxy_type = proxy_type_options[proxy_type_idx].lower()
    address = console_io.PromptResponse('Enter the proxy host address: ')
    log.status.Print()
    if not address:
        return False
    port = console_io.PromptResponse('Enter the proxy port: ')
    log.status.Print()
    if not port:
        return False
    try:
        if not 0 <= int(port) <= 65535:
            log.status.Print('Port number must be between 0 and 65535.')
            return False
    except ValueError:
        log.status.Print('Please enter an integer for the port number.')
        return False
    username, password = (None, None)
    authenticated = console_io.PromptContinue(prompt_string='Is your proxy authenticated', default=False)
    if authenticated:
        username = console_io.PromptResponse('Enter the proxy username: ')
        log.status.Print()
        if not username:
            return False
        password = console_io.PromptResponse('Enter the proxy password: ')
        log.status.Print()
        if not password:
            return False
    SetGcloudProxyProperties(proxy_type=proxy_type, address=address, port=port, username=username, password=password)
    log.status.Print('Cloud SDK proxy properties set.\n')
    return True