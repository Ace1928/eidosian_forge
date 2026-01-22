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
def ChangeGcloudProxySettings():
    """Displays proxy information and helps user set up gcloud proxy properties.

  Returns:
    Whether properties were successfully changed.
  """
    try:
        proxy_info, is_existing_proxy = EffectiveProxyInfo()
    except properties.InvalidValueError:
        log.status.Print('Cloud SDK network proxy settings appear to be invalid. Proxy type, address, and port must be specified. Run [gcloud info] for more details.\n')
        is_existing_proxy = True
    else:
        _DisplayGcloudProxyInfo(proxy_info, is_existing_proxy)
    if properties.VALUES.core.disable_prompts.GetBool():
        return False
    if is_existing_proxy:
        options = ['Change Cloud SDK network proxy properties', 'Clear all gcloud proxy properties', 'Exit']
        existing_proxy_idx = console_io.PromptChoice(options, message='What would you like to do?')
        if existing_proxy_idx == 0:
            return _ProxySetupWalkthrough()
        if existing_proxy_idx == 1:
            SetGcloudProxyProperties()
            log.status.Print('Cloud SDK proxy properties cleared.\n')
            return True
        return False
    else:
        if console_io.PromptContinue(prompt_string='Do you have a network proxy you would like to set in gcloud'):
            return _ProxySetupWalkthrough()
        return False