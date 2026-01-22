import sys

Temporary shim module to indirect the bits of distutils we need from setuptools/distutils while providing useful
error messages beyond `No module named 'distutils' on Python >= 3.12, or when setuptools' vendored distutils is broken.

This is a compromise to avoid a hard-dep on setuptools for Python >= 3.12, since many users don't need runtime compilation support from CFFI.
