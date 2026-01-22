import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
class CCompilerOpt(_Config, _Distutils, _Cache, _CCompiler, _Feature, _Parse):
    """
    A helper class for `CCompiler` aims to provide extra build options
    to effectively control of compiler optimizations that are directly
    related to CPU features.
    """

    def __init__(self, ccompiler, cpu_baseline='min', cpu_dispatch='max', cache_path=None):
        _Config.__init__(self)
        _Distutils.__init__(self, ccompiler)
        _Cache.__init__(self, cache_path, self.dist_info(), cpu_baseline, cpu_dispatch)
        _CCompiler.__init__(self)
        _Feature.__init__(self)
        if not self.cc_noopt and self.cc_has_native:
            self.dist_log("native flag is specified through environment variables. force cpu-baseline='native'")
            cpu_baseline = 'native'
        _Parse.__init__(self, cpu_baseline, cpu_dispatch)
        self._requested_baseline = cpu_baseline
        self._requested_dispatch = cpu_dispatch
        self.sources_status = getattr(self, 'sources_status', {})
        self.cache_private.add('sources_status')
        self.hit_cache = hasattr(self, 'hit_cache')

    def is_cached(self):
        """
        Returns True if the class loaded from the cache file
        """
        return self.cache_infile and self.hit_cache

    def cpu_baseline_flags(self):
        """
        Returns a list of final CPU baseline compiler flags
        """
        return self.parse_baseline_flags

    def cpu_baseline_names(self):
        """
        return a list of final CPU baseline feature names
        """
        return self.parse_baseline_names

    def cpu_dispatch_names(self):
        """
        return a list of final CPU dispatch feature names
        """
        return self.parse_dispatch_names

    def try_dispatch(self, sources, src_dir=None, ccompiler=None, **kwargs):
        """
        Compile one or more dispatch-able sources and generates object files,
        also generates abstract C config headers and macros that
        used later for the final runtime dispatching process.

        The mechanism behind it is to takes each source file that specified
        in 'sources' and branching it into several files depend on
        special configuration statements that must be declared in the
        top of each source which contains targeted CPU features,
        then it compiles every branched source with the proper compiler flags.

        Parameters
        ----------
        sources : list
            Must be a list of dispatch-able sources file paths,
            and configuration statements must be declared inside
            each file.

        src_dir : str
            Path of parent directory for the generated headers and wrapped sources.
            If None(default) the files will generated in-place.

        ccompiler : CCompiler
            Distutils `CCompiler` instance to be used for compilation.
            If None (default), the provided instance during the initialization
            will be used instead.

        **kwargs : any
            Arguments to pass on to the `CCompiler.compile()`

        Returns
        -------
        list : generated object files

        Raises
        ------
        CompileError
            Raises by `CCompiler.compile()` on compiling failure.
        DistutilsError
            Some errors during checking the sanity of configuration statements.

        See Also
        --------
        parse_targets :
            Parsing the configuration statements of dispatch-able sources.
        """
        to_compile = {}
        baseline_flags = self.cpu_baseline_flags()
        include_dirs = kwargs.setdefault('include_dirs', [])
        for src in sources:
            output_dir = os.path.dirname(src)
            if src_dir:
                if not output_dir.startswith(src_dir):
                    output_dir = os.path.join(src_dir, output_dir)
                if output_dir not in include_dirs:
                    include_dirs.append(output_dir)
            has_baseline, targets, extra_flags = self.parse_targets(src)
            nochange = self._generate_config(output_dir, src, targets, has_baseline)
            for tar in targets:
                tar_src = self._wrap_target(output_dir, src, tar, nochange=nochange)
                flags = tuple(extra_flags + self.feature_flags(tar))
                to_compile.setdefault(flags, []).append(tar_src)
            if has_baseline:
                flags = tuple(extra_flags + baseline_flags)
                to_compile.setdefault(flags, []).append(src)
            self.sources_status[src] = (has_baseline, targets)
        objects = []
        for flags, srcs in to_compile.items():
            objects += self.dist_compile(srcs, list(flags), ccompiler=ccompiler, **kwargs)
        return objects

    def generate_dispatch_header(self, header_path):
        """
        Generate the dispatch header which contains the #definitions and headers
        for platform-specific instruction-sets for the enabled CPU baseline and
        dispatch-able features.

        Its highly recommended to take a look at the generated header
        also the generated source files via `try_dispatch()`
        in order to get the full picture.
        """
        self.dist_log('generate CPU dispatch header: (%s)' % header_path)
        baseline_names = self.cpu_baseline_names()
        dispatch_names = self.cpu_dispatch_names()
        baseline_len = len(baseline_names)
        dispatch_len = len(dispatch_names)
        header_dir = os.path.dirname(header_path)
        if not os.path.exists(header_dir):
            self.dist_log(f'dispatch header dir {header_dir} does not exist, creating it', stderr=True)
            os.makedirs(header_dir)
        with open(header_path, 'w') as f:
            baseline_calls = ' \\\n'.join(['\t%sWITH_CPU_EXPAND_(MACRO_TO_CALL(%s, __VA_ARGS__))' % (self.conf_c_prefix, f) for f in baseline_names])
            dispatch_calls = ' \\\n'.join(['\t%sWITH_CPU_EXPAND_(MACRO_TO_CALL(%s, __VA_ARGS__))' % (self.conf_c_prefix, f) for f in dispatch_names])
            f.write(textwrap.dedent('                /*\n                 * AUTOGENERATED DON\'T EDIT\n                 * Please make changes to the code generator (distutils/ccompiler_opt.py)\n                */\n                #define {pfx}WITH_CPU_BASELINE  "{baseline_str}"\n                #define {pfx}WITH_CPU_DISPATCH  "{dispatch_str}"\n                #define {pfx}WITH_CPU_BASELINE_N {baseline_len}\n                #define {pfx}WITH_CPU_DISPATCH_N {dispatch_len}\n                #define {pfx}WITH_CPU_EXPAND_(X) X\n                #define {pfx}WITH_CPU_BASELINE_CALL(MACRO_TO_CALL, ...) \\\n                {baseline_calls}\n                #define {pfx}WITH_CPU_DISPATCH_CALL(MACRO_TO_CALL, ...) \\\n                {dispatch_calls}\n            ').format(pfx=self.conf_c_prefix, baseline_str=' '.join(baseline_names), dispatch_str=' '.join(dispatch_names), baseline_len=baseline_len, dispatch_len=dispatch_len, baseline_calls=baseline_calls, dispatch_calls=dispatch_calls))
            baseline_pre = ''
            for name in baseline_names:
                baseline_pre += self.feature_c_preprocessor(name, tabs=1) + '\n'
            dispatch_pre = ''
            for name in dispatch_names:
                dispatch_pre += textwrap.dedent('                #ifdef {pfx}CPU_TARGET_{name}\n                {pre}\n                #endif /*{pfx}CPU_TARGET_{name}*/\n                ').format(pfx=self.conf_c_prefix_, name=name, pre=self.feature_c_preprocessor(name, tabs=1))
            f.write(textwrap.dedent('            /******* baseline features *******/\n            {baseline_pre}\n            /******* dispatch features *******/\n            {dispatch_pre}\n            ').format(pfx=self.conf_c_prefix_, baseline_pre=baseline_pre, dispatch_pre=dispatch_pre))

    def report(self, full=False):
        report = []
        platform_rows = []
        baseline_rows = []
        dispatch_rows = []
        report.append(('Platform', platform_rows))
        report.append(('', ''))
        report.append(('CPU baseline', baseline_rows))
        report.append(('', ''))
        report.append(('CPU dispatch', dispatch_rows))
        platform_rows.append(('Architecture', 'unsupported' if self.cc_on_noarch else self.cc_march))
        platform_rows.append(('Compiler', 'unix-like' if self.cc_is_nocc else self.cc_name))
        if self.cc_noopt:
            baseline_rows.append(('Requested', 'optimization disabled'))
        else:
            baseline_rows.append(('Requested', repr(self._requested_baseline)))
        baseline_names = self.cpu_baseline_names()
        baseline_rows.append(('Enabled', ' '.join(baseline_names) if baseline_names else 'none'))
        baseline_flags = self.cpu_baseline_flags()
        baseline_rows.append(('Flags', ' '.join(baseline_flags) if baseline_flags else 'none'))
        extra_checks = []
        for name in baseline_names:
            extra_checks += self.feature_extra_checks(name)
        baseline_rows.append(('Extra checks', ' '.join(extra_checks) if extra_checks else 'none'))
        if self.cc_noopt:
            baseline_rows.append(('Requested', 'optimization disabled'))
        else:
            dispatch_rows.append(('Requested', repr(self._requested_dispatch)))
        dispatch_names = self.cpu_dispatch_names()
        dispatch_rows.append(('Enabled', ' '.join(dispatch_names) if dispatch_names else 'none'))
        target_sources = {}
        for source, (_, targets) in self.sources_status.items():
            for tar in targets:
                target_sources.setdefault(tar, []).append(source)
        if not full or not target_sources:
            generated = ''
            for tar in self.feature_sorted(target_sources):
                sources = target_sources[tar]
                name = tar if isinstance(tar, str) else '(%s)' % ' '.join(tar)
                generated += name + '[%d] ' % len(sources)
            dispatch_rows.append(('Generated', generated[:-1] if generated else 'none'))
        else:
            dispatch_rows.append(('Generated', ''))
            for tar in self.feature_sorted(target_sources):
                sources = target_sources[tar]
                pretty_name = tar if isinstance(tar, str) else '(%s)' % ' '.join(tar)
                flags = ' '.join(self.feature_flags(tar))
                implies = ' '.join(self.feature_sorted(self.feature_implies(tar)))
                detect = ' '.join(self.feature_detect(tar))
                extra_checks = []
                for name in (tar,) if isinstance(tar, str) else tar:
                    extra_checks += self.feature_extra_checks(name)
                extra_checks = ' '.join(extra_checks) if extra_checks else 'none'
                dispatch_rows.append(('', ''))
                dispatch_rows.append((pretty_name, implies))
                dispatch_rows.append(('Flags', flags))
                dispatch_rows.append(('Extra checks', extra_checks))
                dispatch_rows.append(('Detect', detect))
                for src in sources:
                    dispatch_rows.append(('', src))
        text = []
        secs_len = [len(secs) for secs, _ in report]
        cols_len = [len(col) for _, rows in report for col, _ in rows]
        tab = ' ' * 2
        pad = max(max(secs_len), max(cols_len))
        for sec, rows in report:
            if not sec:
                text.append('')
                continue
            sec += ' ' * (pad - len(sec))
            text.append(sec + tab + ': ')
            for col, val in rows:
                col += ' ' * (pad - len(col))
                text.append(tab + col + ': ' + val)
        return '\n'.join(text)

    def _wrap_target(self, output_dir, dispatch_src, target, nochange=False):
        assert isinstance(target, (str, tuple))
        if isinstance(target, str):
            ext_name = target_name = target
        else:
            ext_name = '.'.join(target)
            target_name = '__'.join(target)
        wrap_path = os.path.join(output_dir, os.path.basename(dispatch_src))
        wrap_path = '{0}.{2}{1}'.format(*os.path.splitext(wrap_path), ext_name.lower())
        if nochange and os.path.exists(wrap_path):
            return wrap_path
        self.dist_log('wrap dispatch-able target -> ', wrap_path)
        features = self.feature_sorted(self.feature_implies_c(target))
        target_join = '#define %sCPU_TARGET_' % self.conf_c_prefix_
        target_defs = [target_join + f for f in features]
        target_defs = '\n'.join(target_defs)
        with open(wrap_path, 'w') as fd:
            fd.write(textwrap.dedent('            /**\n             * AUTOGENERATED DON\'T EDIT\n             * Please make changes to the code generator              (distutils/ccompiler_opt.py)\n             */\n            #define {pfx}CPU_TARGET_MODE\n            #define {pfx}CPU_TARGET_CURRENT {target_name}\n            {target_defs}\n            #include "{path}"\n            ').format(pfx=self.conf_c_prefix_, target_name=target_name, path=os.path.abspath(dispatch_src), target_defs=target_defs))
        return wrap_path

    def _generate_config(self, output_dir, dispatch_src, targets, has_baseline=False):
        config_path = os.path.basename(dispatch_src)
        config_path = os.path.splitext(config_path)[0] + '.h'
        config_path = os.path.join(output_dir, config_path)
        cache_hash = self.cache_hash(targets, has_baseline)
        try:
            with open(config_path) as f:
                last_hash = f.readline().split('cache_hash:')
                if len(last_hash) == 2 and int(last_hash[1]) == cache_hash:
                    return True
        except OSError:
            pass
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        self.dist_log('generate dispatched config -> ', config_path)
        dispatch_calls = []
        for tar in targets:
            if isinstance(tar, str):
                target_name = tar
            else:
                target_name = '__'.join([t for t in tar])
            req_detect = self.feature_detect(tar)
            req_detect = '&&'.join(['CHK(%s)' % f for f in req_detect])
            dispatch_calls.append('\t%sCPU_DISPATCH_EXPAND_(CB((%s), %s, __VA_ARGS__))' % (self.conf_c_prefix_, req_detect, target_name))
        dispatch_calls = ' \\\n'.join(dispatch_calls)
        if has_baseline:
            baseline_calls = '\t%sCPU_DISPATCH_EXPAND_(CB(__VA_ARGS__))' % self.conf_c_prefix_
        else:
            baseline_calls = ''
        with open(config_path, 'w') as fd:
            fd.write(textwrap.dedent("            // cache_hash:{cache_hash}\n            /**\n             * AUTOGENERATED DON'T EDIT\n             * Please make changes to the code generator (distutils/ccompiler_opt.py)\n             */\n            #ifndef {pfx}CPU_DISPATCH_EXPAND_\n                #define {pfx}CPU_DISPATCH_EXPAND_(X) X\n            #endif\n            #undef {pfx}CPU_DISPATCH_BASELINE_CALL\n            #undef {pfx}CPU_DISPATCH_CALL\n            #define {pfx}CPU_DISPATCH_BASELINE_CALL(CB, ...) \\\n            {baseline_calls}\n            #define {pfx}CPU_DISPATCH_CALL(CHK, CB, ...) \\\n            {dispatch_calls}\n            ").format(pfx=self.conf_c_prefix_, baseline_calls=baseline_calls, dispatch_calls=dispatch_calls, cache_hash=cache_hash))
        return False