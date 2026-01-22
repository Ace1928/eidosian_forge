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
def _compile_with_cache_hip(source, options, arch, cache_dir, extra_source, backend='hiprtc', name_expressions=None, log_stream=None, cache_in_memory=False, use_converter=True):
    global _empty_file_preprocess_cache
    if _is_cudadevrt_needed(options):
        raise ValueError('separate compilation is not supported in HIP')
    options += ('-fcuda-flush-denormals-to-zero',)
    rocm_build_version = driver.get_build_version()
    if rocm_build_version >= 40300000 and rocm_build_version < 40500000:
        options += ('-I' + get_rocm_path() + '/llvm/lib/clang/13.0.0/include/',)
    if cache_dir is None:
        cache_dir = get_cache_dir()
    if arch is None:
        arch = device.Device().compute_capability
    if use_converter:
        source = _convert_to_hip_source(source, extra_source, is_hiprtc=backend == 'hiprtc')
    env = (arch, options, _get_nvrtc_version(), backend)
    base = _empty_file_preprocess_cache.get(env, None)
    if base is None:
        if backend == 'hiprtc':
            base = _preprocess_hiprtc('', options)
        else:
            base = _preprocess_hipcc('', options)
        _empty_file_preprocess_cache[env] = base
    key_src = '%s %s %s %s' % (env, base, source, extra_source)
    key_src = key_src.encode('utf-8')
    name = _hash_hexdigest(key_src) + '.hsaco'
    mod = function.Module()
    if not cache_in_memory:
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, name)
        if os.path.exists(path) and (not name_expressions):
            with open(path, 'rb') as f:
                data = f.read()
            if len(data) >= _hash_length:
                hash_value = data[:_hash_length]
                binary = data[_hash_length:]
                binary_hash = _hash_hexdigest(binary).encode('ascii')
                if hash_value == binary_hash:
                    mod.load(binary)
                    return mod
    else:
        pass
    if backend == 'hiprtc':
        binary, mapping = compile_using_nvrtc(source, options, arch, name + '.cu', name_expressions, log_stream, cache_in_memory)
        mod._set_mapping(mapping)
    else:
        binary = compile_using_hipcc(source, options, arch, log_stream)
    if not cache_in_memory:
        binary_hash = _hash_hexdigest(binary).encode('ascii')
        with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as tf:
            tf.write(binary_hash)
            tf.write(binary)
            temp_path = tf.name
        shutil.move(temp_path, path)
        if _get_bool_env_variable('CUPY_CACHE_SAVE_CUDA_SOURCE', False):
            with open(path + '.cpp', 'w') as f:
                f.write(source)
    else:
        pass
    mod.load(binary)
    return mod