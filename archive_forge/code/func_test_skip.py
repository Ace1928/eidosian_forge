import re, textwrap, os
from os import sys, path
from distutils.errors import DistutilsError
def test_skip(self):
    self.expect('sse vsx neon', x86='sse', ppc64='vsx', armhf='neon', unknown='')
    self.expect('sse41 avx avx2 vsx2 vsx3 neon_vfpv4 asimd', x86='sse41 avx avx2', ppc64='vsx2 vsx3', armhf='neon_vfpv4 asimd', unknown='')
    self.expect('sse neon vsx', baseline='sse neon vsx', x86='', ppc64='', armhf='')
    self.expect('avx2 vsx3 asimdhp', baseline='avx2 vsx3 asimdhp', x86='', ppc64='', armhf='')