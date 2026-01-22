from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.kuberun import kubernetes_consts
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import resource_transform
def GetReadySymbol(ready):
    encoding = console_attr.GetConsoleAttr().GetEncoding()
    if ready == kubernetes_consts.VAL_UNKNOWN:
        return _PickSymbol('…', '.', encoding)
    elif ready == kubernetes_consts.VAL_TRUE or ready == kubernetes_consts.VAL_READY:
        return _PickSymbol('✔', '+', encoding)
    else:
        return 'X'