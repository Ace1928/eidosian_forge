from typing import Dict, List
from .glob_group import GlobGroup, GlobPattern
def _stringify_tree(self, str_list: List[str], preamble: str='', dir_ptr: str='─── '):
    """Recursive method to generate print-friendly version of a Directory."""
    space = '    '
    branch = '│   '
    tee = '├── '
    last = '└── '
    str_list.append(f'{preamble}{dir_ptr}{self.name}\n')
    if dir_ptr == tee:
        preamble = preamble + branch
    else:
        preamble = preamble + space
    file_keys: List[str] = []
    dir_keys: List[str] = []
    for key, val in self.children.items():
        if val.is_dir:
            dir_keys.append(key)
        else:
            file_keys.append(key)
    for index, key in enumerate(sorted(dir_keys)):
        if index == len(dir_keys) - 1 and len(file_keys) == 0:
            self.children[key]._stringify_tree(str_list, preamble, last)
        else:
            self.children[key]._stringify_tree(str_list, preamble, tee)
    for index, file in enumerate(sorted(file_keys)):
        pointer = last if index == len(file_keys) - 1 else tee
        str_list.append(f'{preamble}{pointer}{file}\n')