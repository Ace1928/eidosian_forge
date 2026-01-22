from __future__ import annotations
from .interpreterobjects import extract_required_kwarg
from .. import mlog
from .. import dependencies
from .. import build
from ..wrap import WrapMode
from ..mesonlib import OptionKey, extract_as_list, stringlistify, version_compare_many, listify
from ..dependencies import Dependency, DependencyException, NotFoundDependency
from ..interpreterbase import (MesonInterpreterObject, FeatureNew,
import typing as T
def _handle_featurenew_dependencies(self, name: str) -> None:
    """Do a feature check on dependencies used by this subproject"""
    if name == 'mpi':
        FeatureNew.single_use('MPI Dependency', '0.42.0', self.subproject)
    elif name == 'pcap':
        FeatureNew.single_use('Pcap Dependency', '0.42.0', self.subproject)
    elif name == 'vulkan':
        FeatureNew.single_use('Vulkan Dependency', '0.42.0', self.subproject)
    elif name == 'libwmf':
        FeatureNew.single_use('LibWMF Dependency', '0.44.0', self.subproject)
    elif name == 'openmp':
        FeatureNew.single_use('OpenMP Dependency', '0.46.0', self.subproject)