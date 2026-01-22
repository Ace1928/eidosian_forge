from __future__ import annotations
from mesonbuild.templates.sampleimpl import FileImpl
class FortranProject(FileImpl):
    source_ext = 'f90'
    exe_template = hello_fortran_template
    exe_meson_template = hello_fortran_meson_template
    lib_template = lib_fortran_template
    lib_meson_template = lib_fortran_meson_template
    lib_test_template = lib_fortran_test_template