import re
import os_ken.exception
from os_ken.lib.ofctl_utils import str_to_int
from os_ken.ofproto import nicira_ext
def tokenize_ofp_instruction_arg(arg):
    """
    Tokenize an argument portion of ovs-ofctl style action string.
    """
    arg_re = re.compile('[^,()]*')
    try:
        rest = arg
        result = []
        while len(rest):
            m = arg_re.match(rest)
            if m.end(0) == len(rest):
                result.append(rest)
                return result
            if rest[m.end(0)] == '(':
                this_block, rest = _tokenize_paren_block(rest, m.end(0) + 1)
                result.append(this_block)
            elif rest[m.end(0)] == ',':
                result.append(m.group(0))
                rest = rest[m.end(0):]
            else:
                raise Exception
            if len(rest):
                assert rest[0] == ','
                rest = rest[1:]
        return result
    except Exception:
        raise os_ken.exception.OFPInvalidActionString(action_str=arg)