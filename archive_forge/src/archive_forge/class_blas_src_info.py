import sys
import os
import re
import copy
import warnings
import subprocess
import textwrap
from glob import glob
from functools import reduce
from configparser import NoOptionError
from configparser import RawConfigParser as ConfigParser
from distutils.errors import DistutilsError
from distutils.dist import Distribution
import sysconfig
from numpy.distutils import log
from distutils.util import get_platform
from numpy.distutils.exec_command import (
from numpy.distutils.misc_util import (is_sequence, is_string,
from numpy.distutils.command.config import config as cmd_config
from numpy.distutils import customized_ccompiler as _customized_ccompiler
from numpy.distutils import _shell_utils
import distutils.ccompiler
import tempfile
import shutil
import platform
class blas_src_info(system_info):
    section = 'blas_src'
    dir_env_var = 'BLAS_SRC'
    notfounderror = BlasSrcNotFoundError

    def get_paths(self, section, key):
        pre_dirs = system_info.get_paths(self, section, key)
        dirs = []
        for d in pre_dirs:
            dirs.extend([d] + self.combine_paths(d, ['blas']))
        return [d for d in dirs if os.path.isdir(d)]

    def calc_info(self):
        src_dirs = self.get_src_dirs()
        src_dir = ''
        for d in src_dirs:
            if os.path.isfile(os.path.join(d, 'daxpy.f')):
                src_dir = d
                break
        if not src_dir:
            return
        blas1 = '\n        caxpy csscal dnrm2 dzasum saxpy srotg zdotc ccopy cswap drot\n        dznrm2 scasum srotm zdotu cdotc dasum drotg icamax scnrm2\n        srotmg zdrot cdotu daxpy drotm idamax scopy sscal zdscal crotg\n        dcabs1 drotmg isamax sdot sswap zrotg cscal dcopy dscal izamax\n        snrm2 zaxpy zscal csrot ddot dswap sasum srot zcopy zswap\n        scabs1\n        '
        blas2 = '\n        cgbmv chpmv ctrsv dsymv dtrsv sspr2 strmv zhemv ztpmv cgemv\n        chpr dgbmv dsyr lsame ssymv strsv zher ztpsv cgerc chpr2 dgemv\n        dsyr2 sgbmv ssyr xerbla zher2 ztrmv cgeru ctbmv dger dtbmv\n        sgemv ssyr2 zgbmv zhpmv ztrsv chbmv ctbsv dsbmv dtbsv sger\n        stbmv zgemv zhpr chemv ctpmv dspmv dtpmv ssbmv stbsv zgerc\n        zhpr2 cher ctpsv dspr dtpsv sspmv stpmv zgeru ztbmv cher2\n        ctrmv dspr2 dtrmv sspr stpsv zhbmv ztbsv\n        '
        blas3 = '\n        cgemm csymm ctrsm dsyrk sgemm strmm zhemm zsyr2k chemm csyr2k\n        dgemm dtrmm ssymm strsm zher2k zsyrk cher2k csyrk dsymm dtrsm\n        ssyr2k zherk ztrmm cherk ctrmm dsyr2k ssyrk zgemm zsymm ztrsm\n        '
        sources = [os.path.join(src_dir, f + '.f') for f in (blas1 + blas2 + blas3).split()]
        sources = [f for f in sources if os.path.isfile(f)]
        info = {'sources': sources, 'language': 'f77'}
        self.set_info(**info)