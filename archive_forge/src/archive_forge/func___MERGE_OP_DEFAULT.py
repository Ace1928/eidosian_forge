from __future__ import absolute_import, division, print_function
import json
from copy import (
from difflib import (
def __MERGE_OP_DEFAULT(key_set, command, exist_conf):
    new_conf = exist_conf
    trival_cmd_key_set, dict_list_cmd_key_set = get_key_sets(command)
    nu, dict_list_exist_key_set = get_key_sets(new_conf)
    for key in trival_cmd_key_set:
        new_conf[key] = command[key]
    only_cmd_dict_list_key_set = dict_list_cmd_key_set.difference(dict_list_exist_key_set)
    for key in only_cmd_dict_list_key_set:
        new_conf[key] = command[key]
    return (False, new_conf)