from binascii import unhexlify
from math import ceil
from typing import Any, Dict, List, Tuple, Union, cast
from ._codecs import adobe_glyphs, charset_encoding
from ._utils import b_, logger_error, logger_warning
from .generic import (
def type1_alternative(ft: DictionaryObject, map_dict: Dict[Any, Any], space_code: int, int_entry: List[int]) -> Tuple[Dict[Any, Any], int, List[int]]:
    if '/FontDescriptor' not in ft:
        return (map_dict, space_code, int_entry)
    ft_desc = cast(DictionaryObject, ft['/FontDescriptor']).get('/FontFile')
    if ft_desc is None:
        return (map_dict, space_code, int_entry)
    txt = ft_desc.get_object().get_data()
    txt = txt.split(b'eexec\n')[0]
    txt = txt.split(b'/Encoding')[1]
    lines = txt.replace(b'\r', b'\n').split(b'\n')
    for li in lines:
        if li.startswith(b'dup'):
            words = [_w for _w in li.split(b' ') if _w != b'']
            if len(words) > 3 and words[3] != b'put':
                continue
            try:
                i = int(words[1])
            except ValueError:
                continue
            try:
                v = adobe_glyphs[words[2].decode()]
            except KeyError:
                if words[2].startswith(b'/uni'):
                    try:
                        v = chr(int(words[2][4:], 16))
                    except ValueError:
                        continue
                else:
                    continue
            if words[2].decode() == b' ':
                space_code = i
            map_dict[chr(i)] = v
            int_entry.append(i)
    return (map_dict, space_code, int_entry)