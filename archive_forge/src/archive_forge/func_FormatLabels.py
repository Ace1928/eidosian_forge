from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.kuberun import kubernetesobject
from googlecloudsdk.command_lib.kuberun import kubernetes_consts
from googlecloudsdk.core.console import console_attr
def FormatLabels(labels):
    """Returns a human readable description of user provided labels if any."""
    if not labels:
        return ''
    return ' '.join(sorted(['{}:{}'.format(k, v) for k, v in labels.items() if not k.startswith(kubernetesobject.INTERNAL_GROUPS)]))