from __future__ import annotations
from mesonbuild.templates.sampleimpl import FileHeaderImpl
class ObjCProject(FileHeaderImpl):
    source_ext = 'm'
    header_ext = 'h'
    exe_template = hello_objc_template
    exe_meson_template = hello_objc_meson_template
    lib_template = lib_objc_template
    lib_header_template = lib_h_template
    lib_test_template = lib_objc_test_template
    lib_meson_template = lib_objc_meson_template