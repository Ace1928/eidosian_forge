import re
@staticmethod
def _build_regex_dict(regex_list):
    if regex_list is None:
        return {}
    return dict(((k, re.compile(regex_list[k])) for k in regex_list))