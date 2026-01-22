import os
import socket
from string import Template
from typing import List, Any
class macros:
    """
    Defines simple macros for caffe2.distributed.launch cmd args substitution
    """
    local_rank = '${local_rank}'

    @staticmethod
    def substitute(args: List[Any], local_rank: str) -> List[str]:
        args_sub = []
        for arg in args:
            if isinstance(arg, str):
                sub = Template(arg).safe_substitute(local_rank=local_rank)
                args_sub.append(sub)
            else:
                args_sub.append(arg)
        return args_sub