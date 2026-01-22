import os
import sys
def _add_c_module(dist, ffi, module_name, source, source_extension, kwds):
    from setuptools.command.build_ext import build_ext
    from cffi._shimmed_dist_utils import Extension, log, mkpath
    from cffi import recompiler
    allsources = ['$PLACEHOLDER']
    allsources.extend(kwds.pop('sources', []))
    kwds = _set_py_limited_api(Extension, kwds)
    ext = Extension(name=module_name, sources=allsources, **kwds)

    def make_mod(tmpdir, pre_run=None):
        c_file = os.path.join(tmpdir, module_name + source_extension)
        log.info('generating cffi module %r' % c_file)
        mkpath(tmpdir)
        if pre_run is not None:
            pre_run(ext, ffi)
        updated = recompiler.make_c_source(ffi, module_name, source, c_file)
        if not updated:
            log.info('already up-to-date')
        return c_file
    if dist.ext_modules is None:
        dist.ext_modules = []
    dist.ext_modules.append(ext)
    base_class = dist.cmdclass.get('build_ext', build_ext)

    class build_ext_make_mod(base_class):

        def run(self):
            if ext.sources[0] == '$PLACEHOLDER':
                pre_run = getattr(self, 'pre_run', None)
                ext.sources[0] = make_mod(self.build_temp, pre_run)
            base_class.run(self)
    dist.cmdclass['build_ext'] = build_ext_make_mod