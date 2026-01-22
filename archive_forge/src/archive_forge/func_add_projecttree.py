from __future__ import annotations
import functools, uuid, os, operator
import typing as T
from . import backends
from .. import build
from .. import mesonlib
from .. import mlog
from ..mesonlib import MesonBugException, MesonException, OptionKey
def add_projecttree(self, objects_dict, projecttree_id) -> None:
    root_dict = PbxDict()
    objects_dict.add_item(projecttree_id, root_dict, 'Root of project tree')
    root_dict.add_item('isa', 'PBXGroup')
    target_children = PbxArray()
    root_dict.add_item('children', target_children)
    root_dict.add_item('name', '"Project root"')
    root_dict.add_item('sourceTree', '"<group>"')
    project_tree = self.generate_project_tree()
    self.write_tree(objects_dict, project_tree, target_children, '')