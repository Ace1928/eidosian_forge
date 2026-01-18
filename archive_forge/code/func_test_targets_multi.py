import re, textwrap, os
from os import sys, path
from distutils.errors import DistutilsError
def test_targets_multi(self):
    self.expect_targets('\n            /*@targets\n                (avx512_clx avx512_cnl) (asimdhp asimddp)\n            */\n            ', x86='\\(avx512_clx avx512_cnl\\)', armhf='\\(asimdhp asimddp\\)')
    self.expect_targets('\n            /*@targets\n                f16c (sse41 avx sse42) (sse3 avx2 avx512f)\n                vsx2 (vsx vsx3 vsx2)\n                (neon neon_vfpv4 asimd asimdhp asimddp)\n            */\n            ', x86='avx512f f16c avx', ppc64='vsx3 vsx2', ppc64le='vsx3', armhf='\\(asimdhp asimddp\\)')
    self.expect_targets('\n            /*@targets $keep_sort\n                (sse41 avx sse42) (sse3 avx2 avx512f)\n                (vsx vsx3 vsx2)\n                (asimddp neon neon_vfpv4 asimd asimdhp)\n                (vx vxe vxe2)\n            */\n            ', x86='avx avx512f', ppc64='vsx3', armhf='\\(asimdhp asimddp\\)', s390x='vxe2')
    self.expect_targets('\n            /*@targets $keep_sort\n                fma3 avx2 (fma3 avx2) (avx2 fma3) avx2 fma3\n            */\n            ', x86_gcc='fma3 avx2 \\(fma3 avx2\\)', x86_icc='avx2', x86_iccw='avx2', x86_msvc='avx2')