from __future__ import absolute_import, division, print_function
import os
import re
import logging
import sys
from copy import deepcopy
from ansible.module_utils.basic import env_fallback
def ansible_return(module, rsp, changed, req=None, existing_obj=None, api_context=None):
    """
    :param module: AnsibleModule
    :param rsp: ApiResponse from avi_api
    :param changed: boolean
    :param req: ApiRequest to avi_api
    :param existing_obj: object to be passed debug output
    :param api_context: api login context

    helper function to return the right ansible based on the error code and
    changed
    Returns: specific ansible module exit function
    """
    if rsp is not None and rsp.status_code > 299:
        return module.fail_json(msg='Error %d Msg %s req: %s api_context:%s ' % (rsp.status_code, rsp.text, req, api_context))
    api_creds = AviCredentials()
    api_creds.update_from_ansible_module(module)
    key = '%s:%s:%s' % (api_creds.controller, api_creds.username, api_creds.port)
    disable_fact = module.params.get('avi_disable_session_cache_as_fact')
    fact_context = None
    if not disable_fact:
        fact_context = module.params.get('api_context', {})
        if fact_context:
            fact_context.update({key: api_context})
        else:
            fact_context = {key: api_context}
    obj_val = rsp.json() if rsp else existing_obj
    if obj_val and module.params.get('obj_username', None) and ('username' in obj_val):
        obj_val['obj_username'] = obj_val['username']
    if obj_val and module.params.get('obj_password', None) and ('password' in obj_val):
        obj_val['obj_password'] = obj_val['password']
    old_obj_val = existing_obj if changed and existing_obj else None
    api_context_val = api_context if disable_fact else None
    ansible_facts_val = dict(avi_api_context=fact_context) if not disable_fact else {}
    return module.exit_json(changed=changed, obj=obj_val, old_obj=old_obj_val, ansible_facts=ansible_facts_val, api_context=api_context_val)