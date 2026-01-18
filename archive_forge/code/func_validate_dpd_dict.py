from neutronclient._i18n import _
from neutronclient.common import exceptions
def validate_dpd_dict(dpd_dict):
    for key, value in dpd_dict.items():
        if key not in dpd_supported_keys:
            message = _("DPD Dictionary KeyError: Reason-Invalid DPD key : '%(key)s' not in %(supported_key)s") % {'key': key, 'supported_key': dpd_supported_keys}
            raise exceptions.CommandError(message)
        if key == 'action' and value not in dpd_supported_actions:
            message = _("DPD Dictionary ValueError: Reason-Invalid DPD action : '%(key_value)s' not in %(supported_action)s") % {'key_value': value, 'supported_action': dpd_supported_actions}
            raise exceptions.CommandError(message)
        if key in ('interval', 'timeout'):
            try:
                if int(value) <= 0:
                    raise ValueError()
            except ValueError:
                message = _("DPD Dictionary ValueError: Reason-Invalid positive integer value: '%(key)s' = %(value)s") % {'key': key, 'value': value}
                raise exceptions.CommandError(message)
            else:
                dpd_dict[key] = int(value)
    return