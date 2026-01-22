from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.container import kubeconfig as kconfig
from googlecloudsdk.calliope import base
from googlecloudsdk.core import exceptions as core_exceptions
def GetKubeconfigAndContext(kubeconfig=None, context=None):
    """Get the Kubeconfig path and context."""
    config = kubeconfig or kconfig.Kubeconfig.DefaultPath()
    if not config or not os.access(config, os.R_OK):
        raise MissingConfigError('kubeconfig file not found or is not readable : [{}]'.format(config))
    context_name = context or 'current-context'
    kc = kconfig.Kubeconfig.LoadFromFile(config)
    if context_name == 'current-context':
        context_name = kc.current_context
    elif context_name not in kc.contexts:
        raise ConfigParsingError('context [{}] does not exist in kubeconfig [{}]'.format(context_name, kubeconfig))
    return (config, context_name)