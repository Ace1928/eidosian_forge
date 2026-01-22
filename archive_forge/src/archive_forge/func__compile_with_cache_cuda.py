import copy
import hashlib
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from cupy.cuda import device
from cupy.cuda import function
from cupy.cuda import get_rocm_path
from cupy_backends.cuda.api import driver
from cupy_backends.cuda.api import runtime
from cupy_backends.cuda.libs import nvrtc
from cupy import _util
def _compile_with_cache_cuda(source, options, arch, cache_dir, extra_source=None, backend='nvrtc', enable_cooperative_groups=False, name_expressions=None, log_stream=None, cache_in_memory=False, jitify=False):
    global _empty_file_preprocess_cache
    if cache_dir is None:
        cache_dir = get_cache_dir()
    if arch is None:
        arch = _get_arch()
    options += ('-ftz=true',)
    if enable_cooperative_groups:
        options += ('--device-c',)
    if _get_bool_env_variable('CUPY_CUDA_COMPILE_WITH_DEBUG', False):
        options += ('--device-debug', '--generate-line-info')
    is_jitify_requested = '-DCUPY_USE_JITIFY' in options
    if jitify and (not is_jitify_requested):
        options += ('-DCUPY_USE_JITIFY',)
    elif is_jitify_requested and (not jitify):
        jitify = True
    if jitify and backend != 'nvrtc':
        raise ValueError('jitify only works with NVRTC')
    env = (arch, options, _get_nvrtc_version(), backend) + _get_arch_for_options_for_nvrtc(arch)
    base = _empty_file_preprocess_cache.get(env, None)
    if base is None:
        base = _preprocess('', options, arch, backend)
        _empty_file_preprocess_cache[env] = base
    key_src = '%s %s %s %s' % (env, base, source, extra_source)
    key_src = key_src.encode('utf-8')
    name = _hash_hexdigest(key_src) + '.cubin'
    mod = function.Module()
    if not cache_in_memory:
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, name)
        if os.path.exists(path) and (not name_expressions):
            with open(path, 'rb') as file:
                data = file.read()
            if len(data) >= _hash_length:
                hash = data[:_hash_length]
                cubin = data[_hash_length:]
                cubin_hash = _hash_hexdigest(cubin).encode('ascii')
                if hash == cubin_hash:
                    mod.load(cubin)
                    return mod
    else:
        pass
    if backend == 'nvrtc':
        cu_name = '' if cache_in_memory else name + '.cu'
        ptx, mapping = compile_using_nvrtc(source, options, arch, cu_name, name_expressions, log_stream, cache_in_memory, jitify)
        if _is_cudadevrt_needed(options):
            ls = function.LinkState()
            ls.add_ptr_data(ptx, 'cupy.ptx')
            _cudadevrt = _get_cudadevrt_path()
            ls.add_ptr_file(_cudadevrt)
            cubin = ls.complete()
        else:
            cubin = ptx
        mod._set_mapping(mapping)
    elif backend == 'nvcc':
        rdc = _is_cudadevrt_needed(options)
        cubin = compile_using_nvcc(source, options, arch, name + '.cu', code_type='cubin', separate_compilation=rdc, log_stream=log_stream)
    else:
        raise ValueError('Invalid backend %s' % backend)
    if not cache_in_memory:
        cubin_hash = _hash_hexdigest(cubin).encode('ascii')
        with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as tf:
            tf.write(cubin_hash)
            tf.write(cubin)
            temp_path = tf.name
        shutil.move(temp_path, path)
        if _get_bool_env_variable('CUPY_CACHE_SAVE_CUDA_SOURCE', False):
            with open(path + '.cu', 'w') as f:
                f.write(source)
    else:
        pass
    mod.load(cubin)
    return mod