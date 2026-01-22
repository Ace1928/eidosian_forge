import ctypes as ct
import errno
import os
from pathlib import Path
import platform
from typing import Set, Union
from warnings import warn
import torch
from .env_vars import get_potentially_lib_path_containing_env_vars
class CUDASetup:
    _instance = None

    def __init__(self):
        raise RuntimeError('Call get_instance() instead')

    def generate_instructions(self):
        if getattr(self, 'error', False):
            return
        print(self.error)
        self.error = True
        if not self.cuda_available:
            self.add_log_entry('CUDA SETUP: Problem: The main issue seems to be that the main CUDA library was not detected or CUDA not installed.')
            self.add_log_entry('CUDA SETUP: Solution 1): Your paths are probably not up-to-date. You can update them via: sudo ldconfig.')
            self.add_log_entry('CUDA SETUP: Solution 2): If you do not have sudo rights, you can do the following:')
            self.add_log_entry('CUDA SETUP: Solution 2a): Find the cuda library via: find / -name libcuda.so 2>/dev/null')
            self.add_log_entry('CUDA SETUP: Solution 2b): Once the library is found add it to the LD_LIBRARY_PATH: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:FOUND_PATH_FROM_2a')
            self.add_log_entry('CUDA SETUP: Solution 2c): For a permanent solution add the export from 2b into your .bashrc file, located at ~/.bashrc')
            self.add_log_entry('CUDA SETUP: Solution 3): For a missing CUDA runtime library (libcudart.so), use `find / -name libcudart.so* and follow with step (2b)')
            return
        if self.cudart_path is None:
            self.add_log_entry('CUDA SETUP: Problem: The main issue seems to be that the main CUDA runtime library was not detected.')
            self.add_log_entry('CUDA SETUP: Solution 1: To solve the issue the libcudart.so location needs to be added to the LD_LIBRARY_PATH variable')
            self.add_log_entry('CUDA SETUP: Solution 1a): Find the cuda runtime library via: find / -name libcudart.so 2>/dev/null')
            self.add_log_entry('CUDA SETUP: Solution 1b): Once the library is found add it to the LD_LIBRARY_PATH: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:FOUND_PATH_FROM_1a')
            self.add_log_entry('CUDA SETUP: Solution 1c): For a permanent solution add the export from 1b into your .bashrc file, located at ~/.bashrc')
            self.add_log_entry('CUDA SETUP: Solution 2: If no library was found in step 1a) you need to install CUDA.')
            self.add_log_entry('CUDA SETUP: Solution 2a): Download CUDA install script: wget https://raw.githubusercontent.com/TimDettmers/bitsandbytes/main/cuda_install.sh')
            self.add_log_entry('CUDA SETUP: Solution 2b): Install desired CUDA version to desired location. The syntax is bash cuda_install.sh CUDA_VERSION PATH_TO_INSTALL_INTO.')
            self.add_log_entry('CUDA SETUP: Solution 2b): For example, "bash cuda_install.sh 113 ~/local/" will download CUDA 11.3 and install into the folder ~/local')
            return
        make_cmd = f'CUDA_VERSION={self.cuda_version_string}'
        if len(self.cuda_version_string) < 3:
            make_cmd += ' make cuda92'
        elif self.cuda_version_string == '110':
            make_cmd += ' make cuda110'
        elif self.cuda_version_string[:2] == '11' and int(self.cuda_version_string[2]) > 0:
            make_cmd += ' make cuda11x'
        elif self.cuda_version_string[:2] == '12' and 1 >= int(self.cuda_version_string[2]) >= 0:
            make_cmd += ' make cuda12x'
        elif self.cuda_version_string == '100':
            self.add_log_entry('CUDA SETUP: CUDA 10.0 not supported. Please use a different CUDA version.')
            self.add_log_entry('CUDA SETUP: Before you try again running bitsandbytes, make sure old CUDA 10.0 versions are uninstalled and removed from $LD_LIBRARY_PATH variables.')
            return
        has_cublaslt = is_cublasLt_compatible(self.cc)
        if not has_cublaslt:
            make_cmd += '_nomatmul'
        self.add_log_entry('CUDA SETUP: Something unexpected happened. Please compile from source:')
        self.add_log_entry('git clone https://github.com/TimDettmers/bitsandbytes.git')
        self.add_log_entry('cd bitsandbytes')
        self.add_log_entry(make_cmd)
        self.add_log_entry('python setup.py install')

    def initialize(self):
        if not getattr(self, 'initialized', False):
            self.has_printed = False
            self.lib = None
            self.initialized = False
            self.error = False

    def manual_override(self):
        if not torch.cuda.is_available():
            return
        override_value = os.environ.get('BNB_CUDA_VERSION')
        if not override_value:
            return
        binary_name_stem, _, binary_name_ext = self.binary_name.rpartition('.')
        binary_name_stem = binary_name_stem.rstrip('0123456789')
        self.binary_name = f'{binary_name_stem}{override_value}.{binary_name_ext}'
        warn(f'\n\n{'=' * 80}\nWARNING: Manual override via BNB_CUDA_VERSION env variable detected!\nBNB_CUDA_VERSION=XXX can be used to load a bitsandbytes version that is different from the PyTorch CUDA version.\nIf this was unintended set the BNB_CUDA_VERSION variable to an empty string: export BNB_CUDA_VERSION=\nIf you use the manual override make sure the right libcudart.so is in your LD_LIBRARY_PATH\nFor example by adding the following to your .bashrc: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path_to_cuda_dir/lib64\nLoading: {self.binary_name}\n{'=' * 80}\n\n')

    def run_cuda_setup(self):
        self.initialized = True
        self.cuda_setup_log = []
        binary_name, cudart_path, cc, cuda_version_string = evaluate_cuda_setup()
        self.cudart_path = cudart_path
        self.cuda_available = torch.cuda.is_available()
        self.cc = cc
        self.cuda_version_string = cuda_version_string
        self.binary_name = binary_name
        self.manual_override()
        package_dir = Path(__file__).parent.parent
        binary_path = package_dir / self.binary_name
        try:
            if not binary_path.exists():
                self.add_log_entry(f'CUDA SETUP: Required library version not found: {binary_name}. Maybe you need to compile it from source?')
                legacy_binary_name = f'libbitsandbytes_cpu{DYNAMIC_LIBRARY_SUFFIX}'
                self.add_log_entry(f'CUDA SETUP: Defaulting to {legacy_binary_name}...')
                binary_path = package_dir / legacy_binary_name
                if not binary_path.exists() or torch.cuda.is_available():
                    self.add_log_entry('')
                    self.add_log_entry('=' * 48 + 'ERROR' + '=' * 37)
                    self.add_log_entry('CUDA SETUP: CUDA detection failed! Possible reasons:')
                    self.add_log_entry('1. You need to manually override the PyTorch CUDA version. Please see: "https://github.com/TimDettmers/bitsandbytes/blob/main/how_to_use_nonpytorch_cuda.md')
                    self.add_log_entry('2. CUDA driver not installed')
                    self.add_log_entry('3. CUDA not installed')
                    self.add_log_entry('4. You have multiple conflicting CUDA libraries')
                    self.add_log_entry('5. Required library not pre-compiled for this bitsandbytes release!')
                    self.add_log_entry('CUDA SETUP: If you compiled from source, try again with `make CUDA_VERSION=DETECTED_CUDA_VERSION` for example, `make CUDA_VERSION=118`.')
                    self.add_log_entry('CUDA SETUP: The CUDA version for the compile might depend on your conda install. Inspect CUDA version via `conda list | grep cuda`.')
                    self.add_log_entry('=' * 80)
                    self.add_log_entry('')
                    self.generate_instructions()
                    raise Exception('CUDA SETUP: Setup Failed!')
                self.lib = ct.cdll.LoadLibrary(str(binary_path))
            else:
                self.add_log_entry(f'CUDA SETUP: Loading binary {binary_path!s}...')
                self.lib = ct.cdll.LoadLibrary(str(binary_path))
        except Exception as ex:
            self.add_log_entry(str(ex))

    def add_log_entry(self, msg, is_warning=False):
        self.cuda_setup_log.append((msg, is_warning))

    def print_log_stack(self):
        for msg, is_warning in self.cuda_setup_log:
            if is_warning:
                warn(msg)
            else:
                print(msg)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance