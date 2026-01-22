import os
import sys
import re
from pathlib import Path
from distutils.sysconfig import get_python_lib
from distutils.fancy_getopt import FancyGetopt
from distutils.errors import DistutilsModuleError, \
from distutils.util import split_quoted, strtobool
from numpy.distutils.ccompiler import CCompiler, gen_lib_options
from numpy.distutils import log
from numpy.distutils.misc_util import is_string, all_strings, is_sequence, \
from numpy.distutils.exec_command import find_executable
from numpy.distutils import _shell_utils
from .environment import EnvironmentConfig
class FCompiler(CCompiler):
    """Abstract base class to define the interface that must be implemented
    by real Fortran compiler classes.

    Methods that subclasses may redefine:

        update_executables(), find_executables(), get_version()
        get_flags(), get_flags_opt(), get_flags_arch(), get_flags_debug()
        get_flags_f77(), get_flags_opt_f77(), get_flags_arch_f77(),
        get_flags_debug_f77(), get_flags_f90(), get_flags_opt_f90(),
        get_flags_arch_f90(), get_flags_debug_f90(),
        get_flags_fix(), get_flags_linker_so()

    DON'T call these methods (except get_version) after
    constructing a compiler instance or inside any other method.
    All methods, except update_executables() and find_executables(),
    may call the get_version() method.

    After constructing a compiler instance, always call customize(dist=None)
    method that finalizes compiler construction and makes the following
    attributes available:
      compiler_f77
      compiler_f90
      compiler_fix
      linker_so
      archiver
      ranlib
      libraries
      library_dirs
    """
    distutils_vars = EnvironmentConfig(distutils_section='config_fc', noopt=(None, None, 'noopt', str2bool, False), noarch=(None, None, 'noarch', str2bool, False), debug=(None, None, 'debug', str2bool, False), verbose=(None, None, 'verbose', str2bool, False))
    command_vars = EnvironmentConfig(distutils_section='config_fc', compiler_f77=('exe.compiler_f77', 'F77', 'f77exec', None, False), compiler_f90=('exe.compiler_f90', 'F90', 'f90exec', None, False), compiler_fix=('exe.compiler_fix', 'F90', 'f90exec', None, False), version_cmd=('exe.version_cmd', None, None, None, False), linker_so=('exe.linker_so', 'LDSHARED', 'ldshared', None, False), linker_exe=('exe.linker_exe', 'LD', 'ld', None, False), archiver=(None, 'AR', 'ar', None, False), ranlib=(None, 'RANLIB', 'ranlib', None, False))
    flag_vars = EnvironmentConfig(distutils_section='config_fc', f77=('flags.f77', 'F77FLAGS', 'f77flags', flaglist, True), f90=('flags.f90', 'F90FLAGS', 'f90flags', flaglist, True), free=('flags.free', 'FREEFLAGS', 'freeflags', flaglist, True), fix=('flags.fix', None, None, flaglist, False), opt=('flags.opt', 'FOPT', 'opt', flaglist, True), opt_f77=('flags.opt_f77', None, None, flaglist, False), opt_f90=('flags.opt_f90', None, None, flaglist, False), arch=('flags.arch', 'FARCH', 'arch', flaglist, False), arch_f77=('flags.arch_f77', None, None, flaglist, False), arch_f90=('flags.arch_f90', None, None, flaglist, False), debug=('flags.debug', 'FDEBUG', 'fdebug', flaglist, True), debug_f77=('flags.debug_f77', None, None, flaglist, False), debug_f90=('flags.debug_f90', None, None, flaglist, False), flags=('self.get_flags', 'FFLAGS', 'fflags', flaglist, True), linker_so=('flags.linker_so', 'LDFLAGS', 'ldflags', flaglist, True), linker_exe=('flags.linker_exe', 'LDFLAGS', 'ldflags', flaglist, True), ar=('flags.ar', 'ARFLAGS', 'arflags', flaglist, True))
    language_map = {'.f': 'f77', '.for': 'f77', '.F': 'f77', '.ftn': 'f77', '.f77': 'f77', '.f90': 'f90', '.F90': 'f90', '.f95': 'f90'}
    language_order = ['f90', 'f77']
    compiler_type = None
    compiler_aliases = ()
    version_pattern = None
    possible_executables = []
    executables = {'version_cmd': ['f77', '-v'], 'compiler_f77': ['f77'], 'compiler_f90': ['f90'], 'compiler_fix': ['f90', '-fixed'], 'linker_so': ['f90', '-shared'], 'linker_exe': ['f90'], 'archiver': ['ar', '-cr'], 'ranlib': None}
    suggested_f90_compiler = None
    compile_switch = '-c'
    object_switch = '-o '
    library_switch = '-o '
    module_dir_switch = None
    module_include_switch = '-I'
    pic_flags = []
    src_extensions = ['.for', '.ftn', '.f77', '.f', '.f90', '.f95', '.F', '.F90', '.FOR']
    obj_extension = '.o'
    shared_lib_extension = get_shared_lib_extension()
    static_lib_extension = '.a'
    static_lib_format = 'lib%s%s'
    shared_lib_format = '%s%s'
    exe_extension = ''
    _exe_cache = {}
    _executable_keys = ['version_cmd', 'compiler_f77', 'compiler_f90', 'compiler_fix', 'linker_so', 'linker_exe', 'archiver', 'ranlib']
    c_compiler = None
    extra_f77_compile_args = []
    extra_f90_compile_args = []

    def __init__(self, *args, **kw):
        CCompiler.__init__(self, *args, **kw)
        self.distutils_vars = self.distutils_vars.clone(self._environment_hook)
        self.command_vars = self.command_vars.clone(self._environment_hook)
        self.flag_vars = self.flag_vars.clone(self._environment_hook)
        self.executables = self.executables.copy()
        for e in self._executable_keys:
            if e not in self.executables:
                self.executables[e] = None
        self._is_customised = False

    def __copy__(self):
        obj = self.__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        obj.distutils_vars = obj.distutils_vars.clone(obj._environment_hook)
        obj.command_vars = obj.command_vars.clone(obj._environment_hook)
        obj.flag_vars = obj.flag_vars.clone(obj._environment_hook)
        obj.executables = obj.executables.copy()
        return obj

    def copy(self):
        return self.__copy__()

    def _command_property(key):

        def fget(self):
            assert self._is_customised
            return self.executables[key]
        return property(fget=fget)
    version_cmd = _command_property('version_cmd')
    compiler_f77 = _command_property('compiler_f77')
    compiler_f90 = _command_property('compiler_f90')
    compiler_fix = _command_property('compiler_fix')
    linker_so = _command_property('linker_so')
    linker_exe = _command_property('linker_exe')
    archiver = _command_property('archiver')
    ranlib = _command_property('ranlib')

    def set_executable(self, key, value):
        self.set_command(key, value)

    def set_commands(self, **kw):
        for k, v in kw.items():
            self.set_command(k, v)

    def set_command(self, key, value):
        if not key in self._executable_keys:
            raise ValueError("unknown executable '%s' for class %s" % (key, self.__class__.__name__))
        if is_string(value):
            value = split_quoted(value)
        assert value is None or is_sequence_of_strings(value[1:]), (key, value)
        self.executables[key] = value

    def find_executables(self):
        """Go through the self.executables dictionary, and attempt to
        find and assign appropriate executables.

        Executable names are looked for in the environment (environment
        variables, the distutils.cfg, and command line), the 0th-element of
        the command list, and the self.possible_executables list.

        Also, if the 0th element is "<F77>" or "<F90>", the Fortran 77
        or the Fortran 90 compiler executable is used, unless overridden
        by an environment setting.

        Subclasses should call this if overridden.
        """
        assert self._is_customised
        exe_cache = self._exe_cache

        def cached_find_executable(exe):
            if exe in exe_cache:
                return exe_cache[exe]
            fc_exe = find_executable(exe)
            exe_cache[exe] = exe_cache[fc_exe] = fc_exe
            return fc_exe

        def verify_command_form(name, value):
            if value is not None and (not is_sequence_of_strings(value)):
                raise ValueError('%s value %r is invalid in class %s' % (name, value, self.__class__.__name__))

        def set_exe(exe_key, f77=None, f90=None):
            cmd = self.executables.get(exe_key, None)
            if not cmd:
                return None
            exe_from_environ = getattr(self.command_vars, exe_key)
            if not exe_from_environ:
                possibles = [f90, f77] + self.possible_executables
            else:
                possibles = [exe_from_environ] + self.possible_executables
            seen = set()
            unique_possibles = []
            for e in possibles:
                if e == '<F77>':
                    e = f77
                elif e == '<F90>':
                    e = f90
                if not e or e in seen:
                    continue
                seen.add(e)
                unique_possibles.append(e)
            for exe in unique_possibles:
                fc_exe = cached_find_executable(exe)
                if fc_exe:
                    cmd[0] = fc_exe
                    return fc_exe
            self.set_command(exe_key, None)
            return None
        ctype = self.compiler_type
        f90 = set_exe('compiler_f90')
        if not f90:
            f77 = set_exe('compiler_f77')
            if f77:
                log.warn('%s: no Fortran 90 compiler found' % ctype)
            else:
                raise CompilerNotFound('%s: f90 nor f77' % ctype)
        else:
            f77 = set_exe('compiler_f77', f90=f90)
            if not f77:
                log.warn('%s: no Fortran 77 compiler found' % ctype)
            set_exe('compiler_fix', f90=f90)
        set_exe('linker_so', f77=f77, f90=f90)
        set_exe('linker_exe', f77=f77, f90=f90)
        set_exe('version_cmd', f77=f77, f90=f90)
        set_exe('archiver')
        set_exe('ranlib')

    def update_executables(self):
        """Called at the beginning of customisation. Subclasses should
        override this if they need to set up the executables dictionary.

        Note that self.find_executables() is run afterwards, so the
        self.executables dictionary values can contain <F77> or <F90> as
        the command, which will be replaced by the found F77 or F90
        compiler.
        """
        pass

    def get_flags(self):
        """List of flags common to all compiler types."""
        return [] + self.pic_flags

    def _get_command_flags(self, key):
        cmd = self.executables.get(key, None)
        if cmd is None:
            return []
        return cmd[1:]

    def get_flags_f77(self):
        """List of Fortran 77 specific flags."""
        return self._get_command_flags('compiler_f77')

    def get_flags_f90(self):
        """List of Fortran 90 specific flags."""
        return self._get_command_flags('compiler_f90')

    def get_flags_free(self):
        """List of Fortran 90 free format specific flags."""
        return []

    def get_flags_fix(self):
        """List of Fortran 90 fixed format specific flags."""
        return self._get_command_flags('compiler_fix')

    def get_flags_linker_so(self):
        """List of linker flags to build a shared library."""
        return self._get_command_flags('linker_so')

    def get_flags_linker_exe(self):
        """List of linker flags to build an executable."""
        return self._get_command_flags('linker_exe')

    def get_flags_ar(self):
        """List of archiver flags. """
        return self._get_command_flags('archiver')

    def get_flags_opt(self):
        """List of architecture independent compiler flags."""
        return []

    def get_flags_arch(self):
        """List of architecture dependent compiler flags."""
        return []

    def get_flags_debug(self):
        """List of compiler flags to compile with debugging information."""
        return []
    get_flags_opt_f77 = get_flags_opt_f90 = get_flags_opt
    get_flags_arch_f77 = get_flags_arch_f90 = get_flags_arch
    get_flags_debug_f77 = get_flags_debug_f90 = get_flags_debug

    def get_libraries(self):
        """List of compiler libraries."""
        return self.libraries[:]

    def get_library_dirs(self):
        """List of compiler library directories."""
        return self.library_dirs[:]

    def get_version(self, force=False, ok_status=[0]):
        assert self._is_customised
        version = CCompiler.get_version(self, force=force, ok_status=ok_status)
        if version is None:
            raise CompilerNotFound()
        return version

    def customize(self, dist=None):
        """Customize Fortran compiler.

        This method gets Fortran compiler specific information from
        (i) class definition, (ii) environment, (iii) distutils config
        files, and (iv) command line (later overrides earlier).

        This method should be always called after constructing a
        compiler instance. But not in __init__ because Distribution
        instance is needed for (iii) and (iv).
        """
        log.info('customize %s' % self.__class__.__name__)
        self._is_customised = True
        self.distutils_vars.use_distribution(dist)
        self.command_vars.use_distribution(dist)
        self.flag_vars.use_distribution(dist)
        self.update_executables()
        self.find_executables()
        noopt = self.distutils_vars.get('noopt', False)
        noarch = self.distutils_vars.get('noarch', noopt)
        debug = self.distutils_vars.get('debug', False)
        f77 = self.command_vars.compiler_f77
        f90 = self.command_vars.compiler_f90
        f77flags = []
        f90flags = []
        freeflags = []
        fixflags = []
        if f77:
            f77 = _shell_utils.NativeParser.split(f77)
            f77flags = self.flag_vars.f77
        if f90:
            f90 = _shell_utils.NativeParser.split(f90)
            f90flags = self.flag_vars.f90
            freeflags = self.flag_vars.free
        fix = self.command_vars.compiler_fix
        if fix:
            fix = _shell_utils.NativeParser.split(fix)
            fixflags = self.flag_vars.fix + f90flags
        oflags, aflags, dflags = ([], [], [])

        def get_flags(tag, flags):
            flags.extend(getattr(self.flag_vars, tag))
            this_get = getattr(self, 'get_flags_' + tag)
            for name, c, flagvar in [('f77', f77, f77flags), ('f90', f90, f90flags), ('f90', fix, fixflags)]:
                t = '%s_%s' % (tag, name)
                if c and this_get is not getattr(self, 'get_flags_' + t):
                    flagvar.extend(getattr(self.flag_vars, t))
        if not noopt:
            get_flags('opt', oflags)
            if not noarch:
                get_flags('arch', aflags)
        if debug:
            get_flags('debug', dflags)
        fflags = self.flag_vars.flags + dflags + oflags + aflags
        if f77:
            self.set_commands(compiler_f77=f77 + f77flags + fflags)
        if f90:
            self.set_commands(compiler_f90=f90 + freeflags + f90flags + fflags)
        if fix:
            self.set_commands(compiler_fix=fix + fixflags + fflags)
        linker_so = self.linker_so
        if linker_so:
            linker_so_flags = self.flag_vars.linker_so
            if sys.platform.startswith('aix'):
                python_lib = get_python_lib(standard_lib=1)
                ld_so_aix = os.path.join(python_lib, 'config', 'ld_so_aix')
                python_exp = os.path.join(python_lib, 'config', 'python.exp')
                linker_so = [ld_so_aix] + linker_so + ['-bI:' + python_exp]
            if sys.platform.startswith('os400'):
                from distutils.sysconfig import get_config_var
                python_config = get_config_var('LIBPL')
                ld_so_aix = os.path.join(python_config, 'ld_so_aix')
                python_exp = os.path.join(python_config, 'python.exp')
                linker_so = [ld_so_aix] + linker_so + ['-bI:' + python_exp]
            self.set_commands(linker_so=linker_so + linker_so_flags)
        linker_exe = self.linker_exe
        if linker_exe:
            linker_exe_flags = self.flag_vars.linker_exe
            self.set_commands(linker_exe=linker_exe + linker_exe_flags)
        ar = self.command_vars.archiver
        if ar:
            arflags = self.flag_vars.ar
            self.set_commands(archiver=[ar] + arflags)
        self.set_library_dirs(self.get_library_dirs())
        self.set_libraries(self.get_libraries())

    def dump_properties(self):
        """Print out the attributes of a compiler instance."""
        props = []
        for key in list(self.executables.keys()) + ['version', 'libraries', 'library_dirs', 'object_switch', 'compile_switch']:
            if hasattr(self, key):
                v = getattr(self, key)
                props.append((key, None, '= ' + repr(v)))
        props.sort()
        pretty_printer = FancyGetopt(props)
        for l in pretty_printer.generate_help('%s instance properties:' % self.__class__.__name__):
            if l[:4] == '  --':
                l = '  ' + l[4:]
            print(l)

    def _compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
        """Compile 'src' to product 'obj'."""
        src_flags = {}
        if Path(src).suffix.lower() in FORTRAN_COMMON_FIXED_EXTENSIONS and (not has_f90_header(src)):
            flavor = ':f77'
            compiler = self.compiler_f77
            src_flags = get_f77flags(src)
            extra_compile_args = self.extra_f77_compile_args or []
        elif is_free_format(src):
            flavor = ':f90'
            compiler = self.compiler_f90
            if compiler is None:
                raise DistutilsExecError('f90 not supported by %s needed for %s' % (self.__class__.__name__, src))
            extra_compile_args = self.extra_f90_compile_args or []
        else:
            flavor = ':fix'
            compiler = self.compiler_fix
            if compiler is None:
                raise DistutilsExecError('f90 (fixed) not supported by %s needed for %s' % (self.__class__.__name__, src))
            extra_compile_args = self.extra_f90_compile_args or []
        if self.object_switch[-1] == ' ':
            o_args = [self.object_switch.strip(), obj]
        else:
            o_args = [self.object_switch.strip() + obj]
        assert self.compile_switch.strip()
        s_args = [self.compile_switch, src]
        if extra_compile_args:
            log.info('extra %s options: %r' % (flavor[1:], ' '.join(extra_compile_args)))
        extra_flags = src_flags.get(self.compiler_type, [])
        if extra_flags:
            log.info('using compile options from source: %r' % ' '.join(extra_flags))
        command = compiler + cc_args + extra_flags + s_args + o_args + extra_postargs + extra_compile_args
        display = '%s: %s' % (os.path.basename(compiler[0]) + flavor, src)
        try:
            self.spawn(command, display=display)
        except DistutilsExecError as e:
            msg = str(e)
            raise CompileError(msg) from None

    def module_options(self, module_dirs, module_build_dir):
        options = []
        if self.module_dir_switch is not None:
            if self.module_dir_switch[-1] == ' ':
                options.extend([self.module_dir_switch.strip(), module_build_dir])
            else:
                options.append(self.module_dir_switch.strip() + module_build_dir)
        else:
            print('XXX: module_build_dir=%r option ignored' % module_build_dir)
            print('XXX: Fix module_dir_switch for ', self.__class__.__name__)
        if self.module_include_switch is not None:
            for d in [module_build_dir] + module_dirs:
                options.append('%s%s' % (self.module_include_switch, d))
        else:
            print('XXX: module_dirs=%r option ignored' % module_dirs)
            print('XXX: Fix module_include_switch for ', self.__class__.__name__)
        return options

    def library_option(self, lib):
        return '-l' + lib

    def library_dir_option(self, dir):
        return '-L' + dir

    def link(self, target_desc, objects, output_filename, output_dir=None, libraries=None, library_dirs=None, runtime_library_dirs=None, export_symbols=None, debug=0, extra_preargs=None, extra_postargs=None, build_temp=None, target_lang=None):
        objects, output_dir = self._fix_object_args(objects, output_dir)
        libraries, library_dirs, runtime_library_dirs = self._fix_lib_args(libraries, library_dirs, runtime_library_dirs)
        lib_opts = gen_lib_options(self, library_dirs, runtime_library_dirs, libraries)
        if is_string(output_dir):
            output_filename = os.path.join(output_dir, output_filename)
        elif output_dir is not None:
            raise TypeError("'output_dir' must be a string or None")
        if self._need_link(objects, output_filename):
            if self.library_switch[-1] == ' ':
                o_args = [self.library_switch.strip(), output_filename]
            else:
                o_args = [self.library_switch.strip() + output_filename]
            if is_string(self.objects):
                ld_args = objects + [self.objects]
            else:
                ld_args = objects + self.objects
            ld_args = ld_args + lib_opts + o_args
            if debug:
                ld_args[:0] = ['-g']
            if extra_preargs:
                ld_args[:0] = extra_preargs
            if extra_postargs:
                ld_args.extend(extra_postargs)
            self.mkpath(os.path.dirname(output_filename))
            if target_desc == CCompiler.EXECUTABLE:
                linker = self.linker_exe[:]
            else:
                linker = self.linker_so[:]
            command = linker + ld_args
            try:
                self.spawn(command)
            except DistutilsExecError as e:
                msg = str(e)
                raise LinkError(msg) from None
        else:
            log.debug('skipping %s (up-to-date)', output_filename)

    def _environment_hook(self, name, hook_name):
        if hook_name is None:
            return None
        if is_string(hook_name):
            if hook_name.startswith('self.'):
                hook_name = hook_name[5:]
                hook = getattr(self, hook_name)
                return hook()
            elif hook_name.startswith('exe.'):
                hook_name = hook_name[4:]
                var = self.executables[hook_name]
                if var:
                    return var[0]
                else:
                    return None
            elif hook_name.startswith('flags.'):
                hook_name = hook_name[6:]
                hook = getattr(self, 'get_flags_' + hook_name)
                return hook()
        else:
            return hook_name()

    def can_ccompiler_link(self, ccompiler):
        """
        Check if the given C compiler can link objects produced by
        this compiler.
        """
        return True

    def wrap_unlinkable_objects(self, objects, output_dir, extra_dll_dir):
        """
        Convert a set of object files that are not compatible with the default
        linker, to a file that is compatible.

        Parameters
        ----------
        objects : list
            List of object files to include.
        output_dir : str
            Output directory to place generated object files.
        extra_dll_dir : str
            Output directory to place extra DLL files that need to be
            included on Windows.

        Returns
        -------
        converted_objects : list of str
             List of converted object files.
             Note that the number of output files is not necessarily
             the same as inputs.

        """
        raise NotImplementedError()