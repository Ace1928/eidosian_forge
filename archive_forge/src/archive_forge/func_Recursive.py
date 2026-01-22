from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Dict, Union
from googlecloudsdk.calliope import parser_extensions
def Recursive(level: Level) -> Dict[str, str]:
    ret = {}
    for curr_path, flag_or_level in level.items():
        if isinstance(flag_or_level, str):
            ret[flag_or_level] = curr_path
        else:
            for key, remain_path in Recursive(flag_or_level).items():
                ret[key] = curr_path + '.' + remain_path
    else:
        return ret