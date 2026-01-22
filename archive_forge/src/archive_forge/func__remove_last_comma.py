import json
def _remove_last_comma(str_list, before_index):
    i = before_index - 1
    while str_list[i].isspace() or not str_list[i]:
        i -= 1
    if str_list[i] == ',':
        str_list[i] = ''