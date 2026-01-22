import os
from . import __path__ as _base_path
def _replace_compile_time_constants(shader_source, constants_dict):
    for name, value in constants_dict.items():
        shader_source = shader_source.replace(name, ('%s' % value).encode())
    return shader_source