from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible_collections.community.zabbix.plugins.module_utils.helpers import (
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def user_parameter_difference_check(self, zbx_user, username, name, surname, user_group_ids, passwd, lang, theme, autologin, autologout, refresh, rows_per_page, url, user_medias, timezone, role_name, override_passwd):
    existing_data = copy.deepcopy(zbx_user[0])
    usrgrpids = []
    for usrgrp in existing_data['usrgrps']:
        usrgrpids.append({'usrgrpid': usrgrp['usrgrpid']})
    existing_data['usrgrps'] = sorted(usrgrpids, key=lambda x: x['usrgrpid'])
    existing_data['user_medias'] = existing_data['medias']
    for del_key in ['medias', 'attempt_clock', 'attempt_failed', 'attempt_ip', 'debug_mode', 'users_status', 'gui_access']:
        del existing_data[del_key]
    if 'user_medias' in existing_data and existing_data['user_medias']:
        for user_media in existing_data['user_medias']:
            for del_key in ['mediaid', 'userid']:
                del user_media[del_key]
    request_data = {'userid': zbx_user[0]['userid'], 'username': username, 'name': name, 'surname': surname, 'usrgrps': sorted(user_group_ids, key=lambda x: x['usrgrpid']), 'lang': lang, 'theme': theme, 'autologin': autologin, 'autologout': autologout, 'refresh': refresh, 'rows_per_page': rows_per_page, 'url': url}
    if user_medias:
        request_data['user_medias'] = user_medias
    elif 'user_medias' in existing_data and existing_data['user_medias']:
        del existing_data['user_medias']
    if override_passwd:
        request_data['passwd'] = passwd
    request_data['roleid'] = self.get_roleid_by_name(role_name) if role_name else None
    request_data['timezone'] = timezone
    request_data, del_keys = helper_normalize_data(request_data)
    existing_data, _del_keys = helper_normalize_data(existing_data, del_keys)
    user_parameter_difference_check_result = True
    diff_dict = {}
    if not zabbix_utils.helper_compare_dictionaries(request_data, existing_data, diff_dict):
        user_parameter_difference_check_result = False
    if LooseVersion(self._zbx_api_version) >= LooseVersion('6.4'):
        if user_medias:
            request_data['medias'] = user_medias
            del request_data['user_medias']
    diff_params = {'before': existing_data, 'after': request_data}
    return (user_parameter_difference_check_result, diff_params)