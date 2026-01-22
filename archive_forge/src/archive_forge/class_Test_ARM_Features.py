import sys, platform, re, pytest
from numpy.core._multiarray_umath import (
import numpy as np
import subprocess
import pathlib
import os
import re
@pytest.mark.skipif(not is_linux or not is_arm, reason='Only for Linux and ARM')
class Test_ARM_Features(AbstractTest):
    features = ['NEON', 'ASIMD', 'FPHP', 'ASIMDHP', 'ASIMDDP', 'ASIMDFHM']
    features_groups = dict(NEON_FP16=['NEON', 'HALF'], NEON_VFPV4=['NEON', 'VFPV4'])

    def load_flags(self):
        self.load_flags_cpuinfo('Features')
        arch = self.get_cpuinfo_item('CPU architecture')
        is_rootfs_v8 = int('0' + next(iter(arch))) > 7 if arch else 0
        if re.match('^(aarch64|AARCH64)', machine) or is_rootfs_v8:
            self.features_map = dict(NEON='ASIMD', HALF='ASIMD', VFPV4='ASIMD')
        else:
            self.features_map = dict(ASIMD=('AES', 'SHA1', 'SHA2', 'PMULL', 'CRC32'))