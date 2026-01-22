from __future__ import absolute_import, division, print_function
import json
from copy import (
from difflib import (
def __KEY_MATCH_OP_DEFAULT(key_set, command, exist_conf):
    trival_cmd_key_set, dict_list_cmd_key_set = get_key_sets(command)
    trival_exist_key_set, dict_list_exist_key_set = get_key_sets(exist_conf)
    common_trival_key_set = trival_cmd_key_set.intersection(trival_exist_key_set)
    common_dict_list_key_set = dict_list_cmd_key_set.intersection(dict_list_exist_key_set)
    key_matched_cnt = 0
    for key in common_trival_key_set.union(common_dict_list_key_set):
        if command[key] == exist_conf[key]:
            if key in key_set:
                key_matched_cnt += 1
    key_matched = key_matched_cnt == len(key_set)
    return key_matched