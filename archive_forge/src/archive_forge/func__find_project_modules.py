import os
import re
from parso import python_bytes_to_unicode
from jedi.debug import dbg
from jedi.file_io import KnownContentFileIO, FolderIO
from jedi.inference.names import SubModuleName
from jedi.inference.imports import load_module_from_path
from jedi.inference.filters import ParserTreeFilter
from jedi.inference.gradual.conversion import convert_names
def _find_project_modules(inference_state, module_contexts):
    except_ = [m.py__file__() for m in module_contexts]
    yield from recurse_find_python_files(FolderIO(inference_state.project.path), except_)