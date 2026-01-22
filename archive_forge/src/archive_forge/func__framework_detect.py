from __future__ import annotations
import abc
import re
import os
import typing as T
from .base import DependencyException, DependencyMethods
from .configtool import ConfigToolDependency
from .detect import packages
from .framework import ExtraFrameworkDependency
from .pkgconfig import PkgConfigDependency
from .factory import DependencyFactory
from .. import mlog
from .. import mesonlib
def _framework_detect(self, qvars: T.Dict[str, str], modules: T.List[str], kwargs: T.Dict[str, T.Any]) -> None:
    libdir = qvars['QT_INSTALL_LIBS']
    fw_kwargs = kwargs.copy()
    fw_kwargs.pop('method', None)
    fw_kwargs['paths'] = [libdir]
    for m in modules:
        fname = 'Qt' + m
        mlog.debug('Looking for qt framework ' + fname)
        fwdep = QtExtraFrameworkDependency(fname, self.env, fw_kwargs, qvars, language=self.language)
        if fwdep.found():
            self.compile_args.append('-F' + libdir)
            self.compile_args += fwdep.get_compile_args(with_private_headers=self.private_headers, qt_version=self.version)
            self.link_args += fwdep.get_link_args()
        else:
            self.is_found = False
            break
    else:
        self.is_found = True
        self.bindir = get_qmake_host_bins(qvars)
        self.libexecdir = get_qmake_host_libexecs(qvars)