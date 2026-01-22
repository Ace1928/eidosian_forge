from glob import glob
from distutils.util import get_platform
from distutils.util import convert_path, subst_vars
from distutils.errors import (
from distutils import log, dir_util
from distutils.command.build_scripts import first_line_re
from distutils.spawn import find_executable
from distutils.command import install
import sys
import os
from typing import Dict, List
import zipimport
import shutil
import tempfile
import zipfile
import re
import stat
import random
import textwrap
import warnings
import site
import struct
import contextlib
import subprocess
import shlex
import io
import configparser
import sysconfig
from sysconfig import get_path
from setuptools import Command
from setuptools.sandbox import run_setup
from setuptools.command import setopt
from setuptools.archive_util import unpack_archive
from setuptools.package_index import (
from setuptools.command import bdist_egg, egg_info
from setuptools.warnings import SetuptoolsDeprecationWarning, SetuptoolsWarning
from setuptools.wheel import Wheel
from pkg_resources import (
import pkg_resources
from ..compat import py39, py311
from .._path import ensure_directory
from ..extern.jaraco.text import yield_lines
class easy_install(Command):
    """Manage a download/build/install process"""
    description = 'Find/get/install Python packages'
    command_consumes_arguments = True
    user_options = [('prefix=', None, 'installation prefix'), ('zip-ok', 'z', 'install package as a zipfile'), ('multi-version', 'm', 'make apps have to require() a version'), ('upgrade', 'U', 'force upgrade (searches PyPI for latest versions)'), ('install-dir=', 'd', 'install package to DIR'), ('script-dir=', 's', 'install scripts to DIR'), ('exclude-scripts', 'x', "Don't install scripts"), ('always-copy', 'a', 'Copy all needed packages to install dir'), ('index-url=', 'i', 'base URL of Python Package Index'), ('find-links=', 'f', 'additional URL(s) to search for packages'), ('build-directory=', 'b', 'download/extract/build in DIR; keep the results'), ('optimize=', 'O', 'also compile with optimization: -O1 for "python -O", -O2 for "python -OO", and -O0 to disable [default: -O0]'), ('record=', None, 'filename in which to record list of installed files'), ('always-unzip', 'Z', "don't install as a zipfile, no matter what"), ('site-dirs=', 'S', 'list of directories where .pth files work'), ('editable', 'e', 'Install specified packages in editable form'), ('no-deps', 'N', "don't install dependencies"), ('allow-hosts=', 'H', 'pattern(s) that hostnames must match'), ('local-snapshots-ok', 'l', 'allow building eggs from local checkouts'), ('version', None, 'print version information and exit'), ('no-find-links', None, "Don't load find-links defined in packages being installed"), ('user', None, "install in user site-package '%s'" % site.USER_SITE)]
    boolean_options = ['zip-ok', 'multi-version', 'exclude-scripts', 'upgrade', 'always-copy', 'editable', 'no-deps', 'local-snapshots-ok', 'version', 'user']
    negative_opt = {'always-unzip': 'zip-ok'}
    create_index = PackageIndex

    def initialize_options(self):
        EasyInstallDeprecationWarning.emit()
        self.user = 0
        self.zip_ok = self.local_snapshots_ok = None
        self.install_dir = self.script_dir = self.exclude_scripts = None
        self.index_url = None
        self.find_links = None
        self.build_directory = None
        self.args = None
        self.optimize = self.record = None
        self.upgrade = self.always_copy = self.multi_version = None
        self.editable = self.no_deps = self.allow_hosts = None
        self.root = self.prefix = self.no_report = None
        self.version = None
        self.install_purelib = None
        self.install_platlib = None
        self.install_headers = None
        self.install_lib = None
        self.install_scripts = None
        self.install_data = None
        self.install_base = None
        self.install_platbase = None
        self.install_userbase = site.USER_BASE
        self.install_usersite = site.USER_SITE
        self.no_find_links = None
        self.package_index = None
        self.pth_file = self.always_copy_from = None
        self.site_dirs = None
        self.installed_projects = {}
        self._dry_run = None
        self.verbose = self.distribution.verbose
        self.distribution._set_command_options(self, self.distribution.get_option_dict('easy_install'))

    def delete_blockers(self, blockers):
        extant_blockers = (filename for filename in blockers if os.path.exists(filename) or os.path.islink(filename))
        list(map(self._delete_path, extant_blockers))

    def _delete_path(self, path):
        log.info('Deleting %s', path)
        if self.dry_run:
            return
        is_tree = os.path.isdir(path) and (not os.path.islink(path))
        remover = _rmtree if is_tree else os.unlink
        remover(path)

    @staticmethod
    def _render_version():
        """
        Render the Setuptools version and installation details, then exit.
        """
        ver = '{}.{}'.format(*sys.version_info)
        dist = get_distribution('setuptools')
        tmpl = 'setuptools {dist.version} from {dist.location} (Python {ver})'
        print(tmpl.format(**locals()))
        raise SystemExit()

    def finalize_options(self):
        self.version and self._render_version()
        py_version = sys.version.split()[0]
        self.config_vars = dict(sysconfig.get_config_vars())
        self.config_vars.update({'dist_name': self.distribution.get_name(), 'dist_version': self.distribution.get_version(), 'dist_fullname': self.distribution.get_fullname(), 'py_version': py_version, 'py_version_short': f'{sys.version_info.major}.{sys.version_info.minor}', 'py_version_nodot': f'{sys.version_info.major}{sys.version_info.minor}', 'sys_prefix': self.config_vars['prefix'], 'sys_exec_prefix': self.config_vars['exec_prefix'], 'abiflags': getattr(sys, 'abiflags', ''), 'platlibdir': getattr(sys, 'platlibdir', 'lib')})
        with contextlib.suppress(AttributeError):
            self.config_vars.update({'implementation_lower': install._get_implementation().lower(), 'implementation': install._get_implementation()})
        self.config_vars.setdefault('py_version_nodot_plat', getattr(sys, 'windir', '').replace('.', ''))
        self.config_vars['userbase'] = self.install_userbase
        self.config_vars['usersite'] = self.install_usersite
        if self.user and (not site.ENABLE_USER_SITE):
            log.warn('WARNING: The user site-packages directory is disabled.')
        self._fix_install_dir_for_user_site()
        self.expand_basedirs()
        self.expand_dirs()
        self._expand('install_dir', 'script_dir', 'build_directory', 'site_dirs')
        if self.script_dir is None:
            self.script_dir = self.install_dir
        if self.no_find_links is None:
            self.no_find_links = False
        self.set_undefined_options('install_lib', ('install_dir', 'install_dir'))
        self.set_undefined_options('install_scripts', ('install_dir', 'script_dir'))
        if self.user and self.install_purelib:
            self.install_dir = self.install_purelib
            self.script_dir = self.install_scripts
        self.set_undefined_options('install', ('record', 'record'))
        self.all_site_dirs = get_site_dirs()
        self.all_site_dirs.extend(self._process_site_dirs(self.site_dirs))
        if not self.editable:
            self.check_site_dir()
        default_index = os.getenv('__EASYINSTALL_INDEX', 'https://pypi.org/simple/')
        self.index_url = self.index_url or default_index
        self.shadow_path = self.all_site_dirs[:]
        for path_item in (self.install_dir, normalize_path(self.script_dir)):
            if path_item not in self.shadow_path:
                self.shadow_path.insert(0, path_item)
        if self.allow_hosts is not None:
            hosts = [s.strip() for s in self.allow_hosts.split(',')]
        else:
            hosts = ['*']
        if self.package_index is None:
            self.package_index = self.create_index(self.index_url, search_path=self.shadow_path, hosts=hosts)
        self.local_index = Environment(self.shadow_path + sys.path)
        if self.find_links is not None:
            if isinstance(self.find_links, str):
                self.find_links = self.find_links.split()
        else:
            self.find_links = []
        if self.local_snapshots_ok:
            self.package_index.scan_egg_links(self.shadow_path + sys.path)
        if not self.no_find_links:
            self.package_index.add_find_links(self.find_links)
        self.set_undefined_options('install_lib', ('optimize', 'optimize'))
        self.optimize = self._validate_optimize(self.optimize)
        if self.editable and (not self.build_directory):
            raise DistutilsArgError('Must specify a build directory (-b) when using --editable')
        if not self.args:
            raise DistutilsArgError('No urls, filenames, or requirements specified (see --help)')
        self.outputs = []

    @staticmethod
    def _process_site_dirs(site_dirs):
        if site_dirs is None:
            return
        normpath = map(normalize_path, sys.path)
        site_dirs = [os.path.expanduser(s.strip()) for s in site_dirs.split(',')]
        for d in site_dirs:
            if not os.path.isdir(d):
                log.warn('%s (in --site-dirs) does not exist', d)
            elif normalize_path(d) not in normpath:
                raise DistutilsOptionError(d + ' (in --site-dirs) is not on sys.path')
            else:
                yield normalize_path(d)

    @staticmethod
    def _validate_optimize(value):
        try:
            value = int(value)
            if value not in range(3):
                raise ValueError
        except ValueError as e:
            raise DistutilsOptionError('--optimize must be 0, 1, or 2') from e
        return value

    def _fix_install_dir_for_user_site(self):
        """
        Fix the install_dir if "--user" was used.
        """
        if not self.user:
            return
        self.create_home_path()
        if self.install_userbase is None:
            msg = 'User base directory is not specified'
            raise DistutilsPlatformError(msg)
        self.install_base = self.install_platbase = self.install_userbase
        scheme_name = f'{os.name}_user'
        self.select_scheme(scheme_name)

    def _expand_attrs(self, attrs):
        for attr in attrs:
            val = getattr(self, attr)
            if val is not None:
                if os.name == 'posix' or os.name == 'nt':
                    val = os.path.expanduser(val)
                val = subst_vars(val, self.config_vars)
                setattr(self, attr, val)

    def expand_basedirs(self):
        """Calls `os.path.expanduser` on install_base, install_platbase and
        root."""
        self._expand_attrs(['install_base', 'install_platbase', 'root'])

    def expand_dirs(self):
        """Calls `os.path.expanduser` on install dirs."""
        dirs = ['install_purelib', 'install_platlib', 'install_lib', 'install_headers', 'install_scripts', 'install_data']
        self._expand_attrs(dirs)

    def run(self, show_deprecation=True):
        if show_deprecation:
            self.announce('WARNING: The easy_install command is deprecated and will be removed in a future version.', log.WARN)
        if self.verbose != self.distribution.verbose:
            log.set_verbosity(self.verbose)
        try:
            for spec in self.args:
                self.easy_install(spec, not self.no_deps)
            if self.record:
                outputs = self.outputs
                if self.root:
                    root_len = len(self.root)
                    for counter in range(len(outputs)):
                        outputs[counter] = outputs[counter][root_len:]
                from distutils import file_util
                self.execute(file_util.write_file, (self.record, outputs), "writing list of installed files to '%s'" % self.record)
            self.warn_deprecated_options()
        finally:
            log.set_verbosity(self.distribution.verbose)

    def pseudo_tempname(self):
        """Return a pseudo-tempname base in the install directory.
        This code is intentionally naive; if a malicious party can write to
        the target directory you're already in deep doodoo.
        """
        try:
            pid = os.getpid()
        except Exception:
            pid = random.randint(0, sys.maxsize)
        return os.path.join(self.install_dir, 'test-easy-install-%s' % pid)

    def warn_deprecated_options(self):
        pass

    def check_site_dir(self):
        """Verify that self.install_dir is .pth-capable dir, if needed"""
        instdir = normalize_path(self.install_dir)
        pth_file = os.path.join(instdir, 'easy-install.pth')
        if not os.path.exists(instdir):
            try:
                os.makedirs(instdir)
            except OSError:
                self.cant_write_to_target()
        is_site_dir = instdir in self.all_site_dirs
        if not is_site_dir and (not self.multi_version):
            is_site_dir = self.check_pth_processing()
        else:
            testfile = self.pseudo_tempname() + '.write-test'
            test_exists = os.path.exists(testfile)
            try:
                if test_exists:
                    os.unlink(testfile)
                open(testfile, 'wb').close()
                os.unlink(testfile)
            except OSError:
                self.cant_write_to_target()
        if not is_site_dir and (not self.multi_version):
            pythonpath = os.environ.get('PYTHONPATH', '')
            log.warn(self.__no_default_msg, self.install_dir, pythonpath)
        if is_site_dir:
            if self.pth_file is None:
                self.pth_file = PthDistributions(pth_file, self.all_site_dirs)
        else:
            self.pth_file = None
        if self.multi_version and (not os.path.exists(pth_file)):
            self.pth_file = None
        self.install_dir = instdir
    __cant_write_msg = textwrap.dedent("\n        can't create or remove files in install directory\n\n        The following error occurred while trying to add or remove files in the\n        installation directory:\n\n            %s\n\n        The installation directory you specified (via --install-dir, --prefix, or\n        the distutils default setting) was:\n\n            %s\n        ").lstrip()
    __not_exists_id = textwrap.dedent('\n        This directory does not currently exist.  Please create it and try again, or\n        choose a different installation directory (using the -d or --install-dir\n        option).\n        ').lstrip()
    __access_msg = textwrap.dedent('\n        Perhaps your account does not have write access to this directory?  If the\n        installation directory is a system-owned directory, you may need to sign in\n        as the administrator or "root" account.  If you do not have administrative\n        access to this machine, you may wish to choose a different installation\n        directory, preferably one that is listed in your PYTHONPATH environment\n        variable.\n\n        For information on other options, you may wish to consult the\n        documentation at:\n\n          https://setuptools.pypa.io/en/latest/deprecated/easy_install.html\n\n        Please make the appropriate changes for your system and try again.\n        ').lstrip()

    def cant_write_to_target(self):
        msg = self.__cant_write_msg % (sys.exc_info()[1], self.install_dir)
        if not os.path.exists(self.install_dir):
            msg += '\n' + self.__not_exists_id
        else:
            msg += '\n' + self.__access_msg
        raise DistutilsError(msg)

    def check_pth_processing(self):
        """Empirically verify whether .pth files are supported in inst. dir"""
        instdir = self.install_dir
        log.info('Checking .pth file support in %s', instdir)
        pth_file = self.pseudo_tempname() + '.pth'
        ok_file = pth_file + '.ok'
        ok_exists = os.path.exists(ok_file)
        tmpl = _one_liner('\n            import os\n            f = open({ok_file!r}, \'w\', encoding="utf-8")\n            f.write(\'OK\')\n            f.close()\n            ') + '\n'
        try:
            if ok_exists:
                os.unlink(ok_file)
            dirname = os.path.dirname(ok_file)
            os.makedirs(dirname, exist_ok=True)
            f = open(pth_file, 'w', encoding=py39.LOCALE_ENCODING)
        except OSError:
            self.cant_write_to_target()
        else:
            try:
                f.write(tmpl.format(**locals()))
                f.close()
                f = None
                executable = sys.executable
                if os.name == 'nt':
                    dirname, basename = os.path.split(executable)
                    alt = os.path.join(dirname, 'pythonw.exe')
                    use_alt = basename.lower() == 'python.exe' and os.path.exists(alt)
                    if use_alt:
                        executable = alt
                from distutils.spawn import spawn
                spawn([executable, '-E', '-c', 'pass'], 0)
                if os.path.exists(ok_file):
                    log.info('TEST PASSED: %s appears to support .pth files', instdir)
                    return True
            finally:
                if f:
                    f.close()
                if os.path.exists(ok_file):
                    os.unlink(ok_file)
                if os.path.exists(pth_file):
                    os.unlink(pth_file)
        if not self.multi_version:
            log.warn('TEST FAILED: %s does NOT support .pth files', instdir)
        return False

    def install_egg_scripts(self, dist):
        """Write all the scripts for `dist`, unless scripts are excluded"""
        if not self.exclude_scripts and dist.metadata_isdir('scripts'):
            for script_name in dist.metadata_listdir('scripts'):
                if dist.metadata_isdir('scripts/' + script_name):
                    continue
                self.install_script(dist, script_name, dist.get_metadata('scripts/' + script_name))
        self.install_wrapper_scripts(dist)

    def add_output(self, path):
        if os.path.isdir(path):
            for base, dirs, files in os.walk(path):
                for filename in files:
                    self.outputs.append(os.path.join(base, filename))
        else:
            self.outputs.append(path)

    def not_editable(self, spec):
        if self.editable:
            raise DistutilsArgError("Invalid argument %r: you can't use filenames or URLs with --editable (except via the --find-links option)." % (spec,))

    def check_editable(self, spec):
        if not self.editable:
            return
        if os.path.exists(os.path.join(self.build_directory, spec.key)):
            raise DistutilsArgError("%r already exists in %s; can't do a checkout there" % (spec.key, self.build_directory))

    @contextlib.contextmanager
    def _tmpdir(self):
        tmpdir = tempfile.mkdtemp(prefix='easy_install-')
        try:
            yield str(tmpdir)
        finally:
            os.path.exists(tmpdir) and _rmtree(tmpdir)

    def easy_install(self, spec, deps=False):
        with self._tmpdir() as tmpdir:
            if not isinstance(spec, Requirement):
                if URL_SCHEME(spec):
                    self.not_editable(spec)
                    dl = self.package_index.download(spec, tmpdir)
                    return self.install_item(None, dl, tmpdir, deps, True)
                elif os.path.exists(spec):
                    self.not_editable(spec)
                    return self.install_item(None, spec, tmpdir, deps, True)
                else:
                    spec = parse_requirement_arg(spec)
            self.check_editable(spec)
            dist = self.package_index.fetch_distribution(spec, tmpdir, self.upgrade, self.editable, not self.always_copy, self.local_index)
            if dist is None:
                msg = 'Could not find suitable distribution for %r' % spec
                if self.always_copy:
                    msg += ' (--always-copy skips system and development eggs)'
                raise DistutilsError(msg)
            elif dist.precedence == DEVELOP_DIST:
                self.process_distribution(spec, dist, deps, 'Using')
                return dist
            else:
                return self.install_item(spec, dist.location, tmpdir, deps)

    def install_item(self, spec, download, tmpdir, deps, install_needed=False):
        install_needed = install_needed or self.always_copy
        install_needed = install_needed or os.path.dirname(download) == tmpdir
        install_needed = install_needed or not download.endswith('.egg')
        install_needed = install_needed or (self.always_copy_from is not None and os.path.dirname(normalize_path(download)) == normalize_path(self.always_copy_from))
        if spec and (not install_needed):
            for dist in self.local_index[spec.project_name]:
                if dist.location == download:
                    break
            else:
                install_needed = True
        log.info('Processing %s', os.path.basename(download))
        if install_needed:
            dists = self.install_eggs(spec, download, tmpdir)
            for dist in dists:
                self.process_distribution(spec, dist, deps)
        else:
            dists = [self.egg_distribution(download)]
            self.process_distribution(spec, dists[0], deps, 'Using')
        if spec is not None:
            for dist in dists:
                if dist in spec:
                    return dist
        return None

    def select_scheme(self, name):
        try:
            install._select_scheme(self, name)
        except AttributeError:
            install.install.select_scheme(self, name.replace('posix', 'unix'))

    def process_distribution(self, requirement, dist, deps=True, *info):
        self.update_pth(dist)
        self.package_index.add(dist)
        if dist in self.local_index[dist.key]:
            self.local_index.remove(dist)
        self.local_index.add(dist)
        self.install_egg_scripts(dist)
        self.installed_projects[dist.key] = dist
        log.info(self.installation_report(requirement, dist, *info))
        if dist.has_metadata('dependency_links.txt') and (not self.no_find_links):
            self.package_index.add_find_links(dist.get_metadata_lines('dependency_links.txt'))
        if not deps and (not self.always_copy):
            return
        elif requirement is not None and dist.key != requirement.key:
            log.warn('Skipping dependencies for %s', dist)
            return
        elif requirement is None or dist not in requirement:
            distreq = dist.as_requirement()
            requirement = Requirement(str(distreq))
        log.info('Processing dependencies for %s', requirement)
        try:
            distros = WorkingSet([]).resolve([requirement], self.local_index, self.easy_install)
        except DistributionNotFound as e:
            raise DistutilsError(str(e)) from e
        except VersionConflict as e:
            raise DistutilsError(e.report()) from e
        if self.always_copy or self.always_copy_from:
            for dist in distros:
                if dist.key not in self.installed_projects:
                    self.easy_install(dist.as_requirement())
        log.info('Finished processing dependencies for %s', requirement)

    def should_unzip(self, dist):
        if self.zip_ok is not None:
            return not self.zip_ok
        if dist.has_metadata('not-zip-safe'):
            return True
        if not dist.has_metadata('zip-safe'):
            return True
        return False

    def maybe_move(self, spec, dist_filename, setup_base):
        dst = os.path.join(self.build_directory, spec.key)
        if os.path.exists(dst):
            msg = '%r already exists in %s; build directory %s will not be kept'
            log.warn(msg, spec.key, self.build_directory, setup_base)
            return setup_base
        if os.path.isdir(dist_filename):
            setup_base = dist_filename
        else:
            if os.path.dirname(dist_filename) == setup_base:
                os.unlink(dist_filename)
            contents = os.listdir(setup_base)
            if len(contents) == 1:
                dist_filename = os.path.join(setup_base, contents[0])
                if os.path.isdir(dist_filename):
                    setup_base = dist_filename
        ensure_directory(dst)
        shutil.move(setup_base, dst)
        return dst

    def install_wrapper_scripts(self, dist):
        if self.exclude_scripts:
            return
        for args in ScriptWriter.best().get_args(dist):
            self.write_script(*args)

    def install_script(self, dist, script_name, script_text, dev_path=None):
        """Generate a legacy script wrapper and install it"""
        spec = str(dist.as_requirement())
        is_script = is_python_script(script_text, script_name)
        if is_script:
            body = self._load_template(dev_path) % locals()
            script_text = ScriptWriter.get_header(script_text) + body
        self.write_script(script_name, _to_bytes(script_text), 'b')

    @staticmethod
    def _load_template(dev_path):
        """
        There are a couple of template scripts in the package. This
        function loads one of them and prepares it for use.
        """
        name = 'script.tmpl'
        if dev_path:
            name = name.replace('.tmpl', ' (dev).tmpl')
        raw_bytes = resource_string('setuptools', name)
        return raw_bytes.decode('utf-8')

    def write_script(self, script_name, contents, mode='t', blockers=()):
        """Write an executable file to the scripts directory"""
        self.delete_blockers([os.path.join(self.script_dir, x) for x in blockers])
        log.info('Installing %s script to %s', script_name, self.script_dir)
        target = os.path.join(self.script_dir, script_name)
        self.add_output(target)
        if self.dry_run:
            return
        mask = current_umask()
        ensure_directory(target)
        if os.path.exists(target):
            os.unlink(target)
        with open(target, 'w' + mode) as f:
            f.write(contents)
        chmod(target, 511 - mask)

    def install_eggs(self, spec, dist_filename, tmpdir):
        installer_map = {'.egg': self.install_egg, '.exe': self.install_exe, '.whl': self.install_wheel}
        try:
            install_dist = installer_map[dist_filename.lower()[-4:]]
        except KeyError:
            pass
        else:
            return [install_dist(dist_filename, tmpdir)]
        setup_base = tmpdir
        if os.path.isfile(dist_filename) and (not dist_filename.endswith('.py')):
            unpack_archive(dist_filename, tmpdir, self.unpack_progress)
        elif os.path.isdir(dist_filename):
            setup_base = os.path.abspath(dist_filename)
        if setup_base.startswith(tmpdir) and self.build_directory and (spec is not None):
            setup_base = self.maybe_move(spec, dist_filename, setup_base)
        setup_script = os.path.join(setup_base, 'setup.py')
        if not os.path.exists(setup_script):
            setups = glob(os.path.join(setup_base, '*', 'setup.py'))
            if not setups:
                raise DistutilsError("Couldn't find a setup script in %s" % os.path.abspath(dist_filename))
            if len(setups) > 1:
                raise DistutilsError('Multiple setup scripts in %s' % os.path.abspath(dist_filename))
            setup_script = setups[0]
        if self.editable:
            log.info(self.report_editable(spec, setup_script))
            return []
        else:
            return self.build_and_install(setup_script, setup_base)

    def egg_distribution(self, egg_path):
        if os.path.isdir(egg_path):
            metadata = PathMetadata(egg_path, os.path.join(egg_path, 'EGG-INFO'))
        else:
            metadata = EggMetadata(zipimport.zipimporter(egg_path))
        return Distribution.from_filename(egg_path, metadata=metadata)

    def install_egg(self, egg_path, tmpdir):
        destination = os.path.join(self.install_dir, os.path.basename(egg_path))
        destination = os.path.abspath(destination)
        if not self.dry_run:
            ensure_directory(destination)
        dist = self.egg_distribution(egg_path)
        if not (os.path.exists(destination) and os.path.samefile(egg_path, destination)):
            if os.path.isdir(destination) and (not os.path.islink(destination)):
                dir_util.remove_tree(destination, dry_run=self.dry_run)
            elif os.path.exists(destination):
                self.execute(os.unlink, (destination,), 'Removing ' + destination)
            try:
                new_dist_is_zipped = False
                if os.path.isdir(egg_path):
                    if egg_path.startswith(tmpdir):
                        f, m = (shutil.move, 'Moving')
                    else:
                        f, m = (shutil.copytree, 'Copying')
                elif self.should_unzip(dist):
                    self.mkpath(destination)
                    f, m = (self.unpack_and_compile, 'Extracting')
                else:
                    new_dist_is_zipped = True
                    if egg_path.startswith(tmpdir):
                        f, m = (shutil.move, 'Moving')
                    else:
                        f, m = (shutil.copy2, 'Copying')
                self.execute(f, (egg_path, destination), (m + ' %s to %s') % (os.path.basename(egg_path), os.path.dirname(destination)))
                update_dist_caches(destination, fix_zipimporter_caches=new_dist_is_zipped)
            except Exception:
                update_dist_caches(destination, fix_zipimporter_caches=False)
                raise
        self.add_output(destination)
        return self.egg_distribution(destination)

    def install_exe(self, dist_filename, tmpdir):
        cfg = extract_wininst_cfg(dist_filename)
        if cfg is None:
            raise DistutilsError('%s is not a valid distutils Windows .exe' % dist_filename)
        dist = Distribution(None, project_name=cfg.get('metadata', 'name'), version=cfg.get('metadata', 'version'), platform=get_platform())
        egg_path = os.path.join(tmpdir, dist.egg_name() + '.egg')
        dist.location = egg_path
        egg_tmp = egg_path + '.tmp'
        _egg_info = os.path.join(egg_tmp, 'EGG-INFO')
        pkg_inf = os.path.join(_egg_info, 'PKG-INFO')
        ensure_directory(pkg_inf)
        dist._provider = PathMetadata(egg_tmp, _egg_info)
        self.exe_to_egg(dist_filename, egg_tmp)
        if not os.path.exists(pkg_inf):
            f = open(pkg_inf, 'w')
            f.write('Metadata-Version: 1.0\n')
            for k, v in cfg.items('metadata'):
                if k != 'target_version':
                    f.write('%s: %s\n' % (k.replace('_', '-').title(), v))
            f.close()
        script_dir = os.path.join(_egg_info, 'scripts')
        self.delete_blockers([os.path.join(script_dir, args[0]) for args in ScriptWriter.get_args(dist)])
        bdist_egg.make_zipfile(egg_path, egg_tmp, verbose=self.verbose, dry_run=self.dry_run)
        return self.install_egg(egg_path, tmpdir)

    def exe_to_egg(self, dist_filename, egg_tmp):
        """Extract a bdist_wininst to the directories an egg would use"""
        prefixes = get_exe_prefixes(dist_filename)
        to_compile = []
        native_libs = []
        top_level = {}

        def process(src, dst):
            s = src.lower()
            for old, new in prefixes:
                if s.startswith(old):
                    src = new + src[len(old):]
                    parts = src.split('/')
                    dst = os.path.join(egg_tmp, *parts)
                    dl = dst.lower()
                    if dl.endswith('.pyd') or dl.endswith('.dll'):
                        parts[-1] = bdist_egg.strip_module(parts[-1])
                        top_level[os.path.splitext(parts[0])[0]] = 1
                        native_libs.append(src)
                    elif dl.endswith('.py') and old != 'SCRIPTS/':
                        top_level[os.path.splitext(parts[0])[0]] = 1
                        to_compile.append(dst)
                    return dst
            if not src.endswith('.pth'):
                log.warn("WARNING: can't process %s", src)
            return None
        unpack_archive(dist_filename, egg_tmp, process)
        stubs = []
        for res in native_libs:
            if res.lower().endswith('.pyd'):
                parts = res.split('/')
                resource = parts[-1]
                parts[-1] = bdist_egg.strip_module(parts[-1]) + '.py'
                pyfile = os.path.join(egg_tmp, *parts)
                to_compile.append(pyfile)
                stubs.append(pyfile)
                bdist_egg.write_stub(resource, pyfile)
        self.byte_compile(to_compile)
        bdist_egg.write_safety_flag(os.path.join(egg_tmp, 'EGG-INFO'), bdist_egg.analyze_egg(egg_tmp, stubs))
        for name in ('top_level', 'native_libs'):
            if locals()[name]:
                txt = os.path.join(egg_tmp, 'EGG-INFO', name + '.txt')
                if not os.path.exists(txt):
                    f = open(txt, 'w')
                    f.write('\n'.join(locals()[name]) + '\n')
                    f.close()

    def install_wheel(self, wheel_path, tmpdir):
        wheel = Wheel(wheel_path)
        assert wheel.is_compatible()
        destination = os.path.join(self.install_dir, wheel.egg_name())
        destination = os.path.abspath(destination)
        if not self.dry_run:
            ensure_directory(destination)
        if os.path.isdir(destination) and (not os.path.islink(destination)):
            dir_util.remove_tree(destination, dry_run=self.dry_run)
        elif os.path.exists(destination):
            self.execute(os.unlink, (destination,), 'Removing ' + destination)
        try:
            self.execute(wheel.install_as_egg, (destination,), 'Installing %s to %s' % (os.path.basename(wheel_path), os.path.dirname(destination)))
        finally:
            update_dist_caches(destination, fix_zipimporter_caches=False)
        self.add_output(destination)
        return self.egg_distribution(destination)
    __mv_warning = textwrap.dedent('\n        Because this distribution was installed --multi-version, before you can\n        import modules from this package in an application, you will need to\n        \'import pkg_resources\' and then use a \'require()\' call similar to one of\n        these examples, in order to select the desired version:\n\n            pkg_resources.require("%(name)s")  # latest installed version\n            pkg_resources.require("%(name)s==%(version)s")  # this exact version\n            pkg_resources.require("%(name)s>=%(version)s")  # this version or higher\n        ').lstrip()
    __id_warning = textwrap.dedent("\n        Note also that the installation directory must be on sys.path at runtime for\n        this to work.  (e.g. by being the application's script directory, by being on\n        PYTHONPATH, or by being added to sys.path by your code.)\n        ")

    def installation_report(self, req, dist, what='Installed'):
        """Helpful installation message for display to package users"""
        msg = '\n%(what)s %(eggloc)s%(extras)s'
        if self.multi_version and (not self.no_report):
            msg += '\n' + self.__mv_warning
            if self.install_dir not in map(normalize_path, sys.path):
                msg += '\n' + self.__id_warning
        eggloc = dist.location
        name = dist.project_name
        version = dist.version
        extras = ''
        return msg % locals()
    __editable_msg = textwrap.dedent('\n        Extracted editable version of %(spec)s to %(dirname)s\n\n        If it uses setuptools in its setup script, you can activate it in\n        "development" mode by going to that directory and running::\n\n            %(python)s setup.py develop\n\n        See the setuptools documentation for the "develop" command for more info.\n        ').lstrip()

    def report_editable(self, spec, setup_script):
        dirname = os.path.dirname(setup_script)
        python = sys.executable
        return '\n' + self.__editable_msg % locals()

    def run_setup(self, setup_script, setup_base, args):
        sys.modules.setdefault('distutils.command.bdist_egg', bdist_egg)
        sys.modules.setdefault('distutils.command.egg_info', egg_info)
        args = list(args)
        if self.verbose > 2:
            v = 'v' * (self.verbose - 1)
            args.insert(0, '-' + v)
        elif self.verbose < 2:
            args.insert(0, '-q')
        if self.dry_run:
            args.insert(0, '-n')
        log.info('Running %s %s', setup_script[len(setup_base) + 1:], ' '.join(args))
        try:
            run_setup(setup_script, args)
        except SystemExit as v:
            raise DistutilsError('Setup script exited with %s' % (v.args[0],)) from v

    def build_and_install(self, setup_script, setup_base):
        args = ['bdist_egg', '--dist-dir']
        dist_dir = tempfile.mkdtemp(prefix='egg-dist-tmp-', dir=os.path.dirname(setup_script))
        try:
            self._set_fetcher_options(os.path.dirname(setup_script))
            args.append(dist_dir)
            self.run_setup(setup_script, setup_base, args)
            all_eggs = Environment([dist_dir])
            eggs = []
            for key in all_eggs:
                for dist in all_eggs[key]:
                    eggs.append(self.install_egg(dist.location, setup_base))
            if not eggs and (not self.dry_run):
                log.warn('No eggs found in %s (setup script problem?)', dist_dir)
            return eggs
        finally:
            _rmtree(dist_dir)
            log.set_verbosity(self.verbose)

    def _set_fetcher_options(self, base):
        """
        When easy_install is about to run bdist_egg on a source dist, that
        source dist might have 'setup_requires' directives, requiring
        additional fetching. Ensure the fetcher options given to easy_install
        are available to that command as well.
        """
        ei_opts = self.distribution.get_option_dict('easy_install').copy()
        fetch_directives = ('find_links', 'site_dirs', 'index_url', 'optimize', 'allow_hosts')
        fetch_options = {}
        for key, val in ei_opts.items():
            if key not in fetch_directives:
                continue
            fetch_options[key] = val[1]
        settings = dict(easy_install=fetch_options)
        cfg_filename = os.path.join(base, 'setup.cfg')
        setopt.edit_config(cfg_filename, settings)

    def update_pth(self, dist):
        if self.pth_file is None:
            return
        for d in self.pth_file[dist.key]:
            if not self.multi_version and d.location == dist.location:
                continue
            log.info('Removing %s from easy-install.pth file', d)
            self.pth_file.remove(d)
            if d.location in self.shadow_path:
                self.shadow_path.remove(d.location)
        if not self.multi_version:
            if dist.location in self.pth_file.paths:
                log.info('%s is already the active version in easy-install.pth', dist)
            else:
                log.info('Adding %s to easy-install.pth file', dist)
                self.pth_file.add(dist)
                if dist.location not in self.shadow_path:
                    self.shadow_path.append(dist.location)
        if self.dry_run:
            return
        self.pth_file.save()
        if dist.key != 'setuptools':
            return
        filename = os.path.join(self.install_dir, 'setuptools.pth')
        if os.path.islink(filename):
            os.unlink(filename)
        with open(filename, 'wt', encoding=py39.LOCALE_ENCODING) as f:
            f.write(self.pth_file.make_relative(dist.location) + '\n')

    def unpack_progress(self, src, dst):
        log.debug('Unpacking %s to %s', src, dst)
        return dst

    def unpack_and_compile(self, egg_path, destination):
        to_compile = []
        to_chmod = []

        def pf(src, dst):
            if dst.endswith('.py') and (not src.startswith('EGG-INFO/')):
                to_compile.append(dst)
            elif dst.endswith('.dll') or dst.endswith('.so'):
                to_chmod.append(dst)
            self.unpack_progress(src, dst)
            return not self.dry_run and dst or None
        unpack_archive(egg_path, destination, pf)
        self.byte_compile(to_compile)
        if not self.dry_run:
            for f in to_chmod:
                mode = (os.stat(f)[stat.ST_MODE] | 365) & 4077
                chmod(f, mode)

    def byte_compile(self, to_compile):
        if sys.dont_write_bytecode:
            return
        from distutils.util import byte_compile
        try:
            log.set_verbosity(self.verbose - 1)
            byte_compile(to_compile, optimize=0, force=1, dry_run=self.dry_run)
            if self.optimize:
                byte_compile(to_compile, optimize=self.optimize, force=1, dry_run=self.dry_run)
        finally:
            log.set_verbosity(self.verbose)
    __no_default_msg = textwrap.dedent('\n        bad install directory or PYTHONPATH\n\n        You are attempting to install a package to a directory that is not\n        on PYTHONPATH and which Python does not read ".pth" files from.  The\n        installation directory you specified (via --install-dir, --prefix, or\n        the distutils default setting) was:\n\n            %s\n\n        and your PYTHONPATH environment variable currently contains:\n\n            %r\n\n        Here are some of your options for correcting the problem:\n\n        * You can choose a different installation directory, i.e., one that is\n          on PYTHONPATH or supports .pth files\n\n        * You can add the installation directory to the PYTHONPATH environment\n          variable.  (It must then also be on PYTHONPATH whenever you run\n          Python and want to use the package(s) you are installing.)\n\n        * You can set up the installation directory to support ".pth" files by\n          using one of the approaches described here:\n\n          https://setuptools.pypa.io/en/latest/deprecated/easy_install.html#custom-installation-locations\n\n\n        Please make the appropriate changes for your system and try again.\n        ').strip()

    def create_home_path(self):
        """Create directories under ~."""
        if not self.user:
            return
        home = convert_path(os.path.expanduser('~'))
        for path in only_strs(self.config_vars.values()):
            if path.startswith(home) and (not os.path.isdir(path)):
                self.debug_print("os.makedirs('%s', 0o700)" % path)
                os.makedirs(path, 448)
    INSTALL_SCHEMES = dict(posix=dict(install_dir='$base/lib/python$py_version_short/site-packages', script_dir='$base/bin'))
    DEFAULT_SCHEME = dict(install_dir='$base/Lib/site-packages', script_dir='$base/Scripts')

    def _expand(self, *attrs):
        config_vars = self.get_finalized_command('install').config_vars
        if self.prefix:
            config_vars = dict(config_vars)
            config_vars['base'] = self.prefix
            scheme = self.INSTALL_SCHEMES.get(os.name, self.DEFAULT_SCHEME)
            for attr, val in scheme.items():
                if getattr(self, attr, None) is None:
                    setattr(self, attr, val)
        from distutils.util import subst_vars
        for attr in attrs:
            val = getattr(self, attr)
            if val is not None:
                val = subst_vars(val, config_vars)
                if os.name == 'posix':
                    val = os.path.expanduser(val)
                setattr(self, attr, val)