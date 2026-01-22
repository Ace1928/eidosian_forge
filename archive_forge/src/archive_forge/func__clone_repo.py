import gc
from io import BytesIO
import logging
import os
import os.path as osp
import stat
import uuid
import git
from git.cmd import Git
from git.compat import defenc
from git.config import GitConfigParser, SectionConstraint, cp
from git.exc import (
from git.objects.base import IndexObject, Object
from git.objects.util import TraversableIterableObj
from git.util import (
from .util import (
from typing import Callable, Dict, Mapping, Sequence, TYPE_CHECKING, cast
from typing import Any, Iterator, Union
from git.types import Commit_ish, Literal, PathLike, TBD
@classmethod
def _clone_repo(cls, repo: 'Repo', url: str, path: PathLike, name: str, allow_unsafe_options: bool=False, allow_unsafe_protocols: bool=False, **kwargs: Any) -> 'Repo':
    """
        :return: Repo instance of newly cloned repository
        :param repo: Our parent repository
        :param url: URL to clone from
        :param path: Repository - relative path to the submodule checkout location
        :param name: Canonical name of the submodule
        :param allow_unsafe_protocols: Allow unsafe protocols to be used, like ext
        :param allow_unsafe_options: Allow unsafe options to be used, like --upload-pack
        :param kwargs: Additional arguments given to git.clone
        """
    module_abspath = cls._module_abspath(repo, path, name)
    module_checkout_path = module_abspath
    if cls._need_gitfile_submodules(repo.git):
        kwargs['separate_git_dir'] = module_abspath
        module_abspath_dir = osp.dirname(module_abspath)
        if not osp.isdir(module_abspath_dir):
            os.makedirs(module_abspath_dir)
        module_checkout_path = osp.join(str(repo.working_tree_dir), path)
    clone = git.Repo.clone_from(url, module_checkout_path, allow_unsafe_options=allow_unsafe_options, allow_unsafe_protocols=allow_unsafe_protocols, **kwargs)
    if cls._need_gitfile_submodules(repo.git):
        cls._write_git_file_and_module_config(module_checkout_path, module_abspath)
    return clone