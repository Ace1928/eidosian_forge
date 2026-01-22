import argparse
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
class Libdevice(ExternLibrary):
    _symbol_groups: Dict[str, List[Symbol]]

    def __init__(self, path) -> None:
        """
        Constructor for Libdevice.
        :param path: path of the libdevice library
        """
        super().__init__('libdevice', path)
        self._symbol_groups = {}
        self.is_pure = True

    @staticmethod
    def _extract_symbol(line) -> Optional[Symbol]:
        entries = line.split('@')
        ret_str = entries[0]
        func_str = entries[1]
        ret_strs = ret_str.split()
        if ret_strs[1] == 'internal':
            return None
        ret_type = convert_type(ret_strs[1])
        if ret_type is None:
            return None
        func_strs = func_str.split('(')
        func_name = func_strs[0].replace('@', '')
        op_name = func_name.replace('__nv_', '')
        if 'ieee' in op_name:
            return None
        arg_strs = func_strs[1].split(',')
        arg_types = []
        arg_names = []
        for i, arg_str in enumerate(arg_strs):
            arg_type = convert_type(arg_str.split()[0])
            if arg_type is None:
                return None
            arg_name = 'arg' + str(i)
            arg_types.append(arg_type)
            arg_names.append(arg_name)
        if op_name == 'sad':
            arg_types[-1] = to_unsigned(arg_types[-1])
        elif op_name.startswith('u'):
            ret_type = to_unsigned(ret_type)
            for i, arg_type in enumerate(arg_types):
                arg_types[i] = to_unsigned(arg_type)
        return Symbol(func_name, op_name, ret_type, arg_names, arg_types)

    def _group_symbols(self) -> None:
        symbol_set = {}
        for symbol in self._symbols.values():
            op_name = symbol.op_name
            symbol_set[op_name] = symbol
        renaming = {'llabs': 'abs', 'acosf': 'acos', 'acoshf': 'acosh', 'dadd_rd': 'add_rd', 'fadd_rd': 'add_rd', 'dadd_rn': 'add_rn', 'fadd_rn': 'add_rn', 'dadd_ru': 'add_ru', 'fadd_ru': 'add_ru', 'dadd_rz': 'add_rz', 'fadd_rz': 'add_rz', 'asinf': 'asin', 'asinhf': 'asinh', 'atanf': 'atan', 'atan2f': 'atan2', 'atanhf': 'atanh', 'brevll': 'brev', 'cbrtf': 'cbrt', 'ceilf': 'ceil', 'clzll': 'clz', 'copysignf': 'copysign', 'cosf': 'cos', 'coshf': 'cosh', 'cospif': 'cospi', 'cyl_bessel_i0f': 'cyl_bessel_i0', 'cyl_bessel_i1f': 'cyl_bessel_i1', 'fdiv_rd': 'div_rd', 'ddiv_rd': 'div_rd', 'fdiv_rn': 'div_rn', 'ddiv_rn': 'div_rn', 'fdiv_ru': 'div_ru', 'ddiv_ru': 'div_ru', 'fdiv_rz': 'div_rz', 'ddiv_rz': 'div_rz', 'erff': 'erf', 'erfcf': 'erfc', 'erfcinvf': 'erfcinv', 'erfcxf': 'erfcx', 'erfinvf': 'erfinv', 'expf': 'exp', 'exp10f': 'exp10', 'exp2f': 'exp2', 'expm1f': 'expm1', 'fabsf': 'abs', 'fabs': 'abs', 'fast_fdividef': 'fast_dividef', 'fdimf': 'fdim', 'ffsll': 'ffs', 'floorf': 'floor', 'fmaf': 'fma', 'fmaf_rd': 'fma_rd', 'fmaf_rn': 'fma_rn', 'fmaf_ru': 'fma_ru', 'fmaf_rz': 'fma_rz', 'fmodf': 'fmod', 'uhadd': 'hadd', 'hypotf': 'hypot', 'ilogbf': 'ilogb', 'isinff': 'isinf', 'isinfd': 'isinf', 'isnanf': 'isnan', 'isnand': 'isnan', 'j0f': 'j0', 'j1f': 'j1', 'jnf': 'jn', 'ldexpf': 'ldexp', 'lgammaf': 'lgamma', 'llrintf': 'llrint', 'llroundf': 'llround', 'logf': 'log', 'log10f': 'log10', 'log1pf': 'log1p', 'log2f': 'log2', 'logbf': 'logb', 'umax': 'max', 'llmax': 'max', 'ullmax': 'max', 'fmaxf': 'max', 'fmax': 'max', 'umin': 'min', 'llmin': 'min', 'ullmin': 'min', 'fminf': 'min', 'fmin': 'min', 'dmul_rd': 'mul_rd', 'fmul_rd': 'mul_rd', 'dmul_rn': 'mul_rn', 'fmul_rn': 'mul_rn', 'dmul_ru': 'mul_ru', 'fmul_ru': 'mul_ru', 'dmul_rz': 'mul_rz', 'fmul_rz': 'mul_rz', 'umul24': 'mul24', 'umulhi': 'mulhi', 'mul64hi': 'mulhi', 'umul64hi': 'mulhi', 'nearbyintf': 'nearbyint', 'nextafterf': 'nextafter', 'norm3df': 'norm3d', 'norm4df': 'norm4d', 'normcdff': 'normcdf', 'normcdfinvf': 'normcdfinv', 'popcll': 'popc', 'powif': 'pow', 'powi': 'pow', 'powf': 'pow', 'rcbrtf': 'rcbrt', 'frcp_rd': 'rcp_rd', 'drcp_rd': 'rcp_rd', 'frcp_rn': 'rcp_rn', 'drcp_rn': 'rcp_rn', 'frcp_ru': 'rcp_ru', 'drcp_ru': 'rcp_ru', 'frcp_rz': 'rcp_rz', 'drcp_rz': 'rcp_rz', 'remainderf': 'remainder', 'urhadd': 'rhadd', 'rhypotf': 'rhypot', 'rintf': 'rint', 'rnorm3df': 'rnorm3d', 'rnorm4df': 'rnorm4d', 'roundf': 'round', 'rsqrtf': 'rsqrt', 'frsqrt_rn': 'rsqrt_rn', 'usad': 'sad', 'scalbnf': 'scalbn', 'signbitf': 'signbit', 'signbitd': 'signbit', 'sinf': 'sin', 'sinhf': 'sinh', 'sinpif': 'sinpi', 'sqrtf': 'sqrt', 'fsqrt_rd': 'sqrt_rd', 'dsqrt_rd': 'sqrt_rd', 'fsqrt_rn': 'sqrt_rn', 'dsqrt_rn': 'sqrt_rn', 'fsqrt_ru': 'sqrt_ru', 'dsqrt_ru': 'sqrt_ru', 'fsqrt_rz': 'sqrt_rz', 'dsqrt_rz': 'sqrt_rz', 'fsub_rd': 'sub_rd', 'dsub_rd': 'sub_rd', 'fsub_rn': 'sub_rn', 'dsub_rn': 'sub_rn', 'fsub_ru': 'sub_ru', 'dsub_ru': 'sub_ru', 'fsub_rz': 'sub_rz', 'dsub_rz': 'sub_rz', 'tanf': 'tan', 'tanhf': 'tanh', 'tgammaf': 'tgamma', 'truncf': 'trunc', 'y0f': 'y0', 'y1f': 'y1', 'ynf': 'yn'}
        for symbol in self._symbols.values():
            op_name = symbol.op_name
            if op_name in renaming:
                op_name = renaming[op_name]
                symbol._op_name = op_name
            if op_name in self._symbol_groups:
                self._symbol_groups[op_name].append(symbol)
            else:
                self._symbol_groups[op_name] = [symbol]

    def parse_symbols(self, input_file) -> None:
        if len(self.symbols) > 0:
            return
        output = subprocess.check_output(['grep', 'define', input_file]).decode().splitlines()
        for line in output:
            symbol = self._extract_symbol(line)
            if symbol is None:
                continue
            self._symbols[symbol.name] = symbol
        self._group_symbols()

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