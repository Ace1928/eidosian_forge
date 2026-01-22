import collections
import inspect
from neutron_lib._i18n import _
from neutron_lib.callbacks import manager
from neutron_lib.callbacks import priority_group
def _get_callback_manager():
    global _CALLBACK_MANAGER
    if _CALLBACK_MANAGER is None:
        _CALLBACK_MANAGER = manager.CallbacksManager()
    return _CALLBACK_MANAGER