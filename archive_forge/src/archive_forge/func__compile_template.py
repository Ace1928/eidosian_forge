import atexit
import os
import re
import shutil
import textwrap
import threading
from typing import Any, List, Optional
import torch
from torch.utils.benchmark.utils._stubs import CallgrindModuleType, TimeitModuleType
from torch.utils.benchmark.utils.common import _make_temp_dir
from torch.utils import cpp_extension
def _compile_template(*, stmt: str, setup: str, global_setup: str, src: str, is_standalone: bool) -> Any:
    for before, after, indentation in (('// GLOBAL_SETUP_TEMPLATE_LOCATION', global_setup, 0), ('// SETUP_TEMPLATE_LOCATION', setup, 4), ('// STMT_TEMPLATE_LOCATION', stmt, 8)):
        src = re.sub(before, textwrap.indent(after, ' ' * indentation)[indentation:], src)
    with LOCK:
        name = f'timer_cpp_{abs(hash(src))}'
        build_dir = os.path.join(_get_build_root(), name)
        os.makedirs(build_dir, exist_ok=True)
        src_path = os.path.join(build_dir, 'timer_src.cpp')
        with open(src_path, 'w') as f:
            f.write(src)
    return cpp_extension.load(name=name, sources=[src_path], build_directory=build_dir, extra_cflags=CXX_FLAGS, extra_include_paths=EXTRA_INCLUDE_PATHS, is_python_module=not is_standalone, is_standalone=is_standalone)