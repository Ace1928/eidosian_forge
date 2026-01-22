import abc
from neutron_lib._i18n import _
from neutron_lib import constants
@classmethod
def _assert_api_definition(cls, attr=None):
    if cls.api_definition == _UNSET:
        raise NotImplementedError(_('Extension module API definition not set.'))
    if attr and getattr(cls.api_definition, attr, _UNSET) == _UNSET:
        raise NotImplementedError(_("Extension module API definition does not define '%s'") % attr)