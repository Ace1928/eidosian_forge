from time import sleep
from time import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
def _set_kwarg(kwargs, key, value):
    mapped_key = PARAMS_MAP[key]
    if '/' in mapped_key:
        key_list = mapped_key.split('/')
        key_list.reverse()
    else:
        key_list = [mapped_key]
    data = kwargs
    while len(key_list) > 1:
        this_key = key_list.pop()
        if this_key not in data:
            data[this_key] = {}
        data = data[this_key]
    data[key_list[0]] = value