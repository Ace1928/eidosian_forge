import argparse
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
def _output_stubs(self) -> str:
    import_str = 'from . import core\n'
    import_str += 'import os\n'
    import_str += 'import functools\n'
    header_str = ''
    header_str += '@functools.lru_cache()\n'
    header_str += 'def libdevice_path():\n'
    header_str += '    import torch\n'
    header_str += '    third_party_dir =  os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party")\n'
    header_str += '    if torch.version.hip is None:\n'
    header_str += '        default = os.path.join(third_party_dir, "cuda", "lib", "libdevice.10.bc")\n'
    header_str += '    else:\n'
    header_str += "        default = ''\n"
    header_str += '    return os.getenv("TRITON_LIBDEVICE_PATH", default)\n'
    func_str = ''
    for symbols in self._symbol_groups.values():
        func_str += '@core.extern\n'
        func_name_str = f'def {symbols[0].op_name}('
        for arg_name in symbols[0].arg_names:
            func_name_str += f'{arg_name}, '
        func_name_str += '_builder=None):\n'
        return_str = f'\treturn core.extern_elementwise("{self._name}", libdevice_path(), ['
        for arg_name in symbols[0].arg_names:
            return_str += f'{arg_name}, '
        return_str += '], \n'
        arg_type_symbol_dict_str = '{'
        for symbol in symbols:
            arg_type_symbol_dict_str += '('
            for arg_type in symbol.arg_types:
                arg_type_symbol_dict_str += f'core.dtype("{arg_type}"),'
            ret_type = f'core.dtype("{symbol.ret_type}")'
            arg_type_symbol_dict_str += '): ("' + symbol.name + '", ' + ret_type + '),\n'
        arg_type_symbol_dict_str += '}'
        return_str += arg_type_symbol_dict_str
        return_str += f', is_pure={self.is_pure}'
        return_str += ', _builder=_builder)\n'
        func_str += func_name_str + return_str + '\n'
    file_str = import_str + header_str + func_str
    return file_str