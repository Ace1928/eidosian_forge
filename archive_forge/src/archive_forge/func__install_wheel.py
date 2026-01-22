import collections
import compileall
import contextlib
import csv
import importlib
import logging
import os.path
import re
import shutil
import sys
import warnings
from base64 import urlsafe_b64encode
from email.message import Message
from itertools import chain, filterfalse, starmap
from typing import (
from zipfile import ZipFile, ZipInfo
from pip._vendor.distlib.scripts import ScriptMaker
from pip._vendor.distlib.util import get_export_entry
from pip._vendor.packaging.utils import canonicalize_name
from pip._internal.exceptions import InstallationError
from pip._internal.locations import get_major_minor_version
from pip._internal.metadata import (
from pip._internal.models.direct_url import DIRECT_URL_METADATA_NAME, DirectUrl
from pip._internal.models.scheme import SCHEME_KEYS, Scheme
from pip._internal.utils.filesystem import adjacent_tmp_file, replace
from pip._internal.utils.misc import captured_stdout, ensure_dir, hash_file, partition
from pip._internal.utils.unpacking import (
from pip._internal.utils.wheel import parse_wheel
def _install_wheel(name: str, wheel_zip: ZipFile, wheel_path: str, scheme: Scheme, pycompile: bool=True, warn_script_location: bool=True, direct_url: Optional[DirectUrl]=None, requested: bool=False) -> None:
    """Install a wheel.

    :param name: Name of the project to install
    :param wheel_zip: open ZipFile for wheel being installed
    :param scheme: Distutils scheme dictating the install directories
    :param req_description: String used in place of the requirement, for
        logging
    :param pycompile: Whether to byte-compile installed Python files
    :param warn_script_location: Whether to check that scripts are installed
        into a directory on PATH
    :raises UnsupportedWheel:
        * when the directory holds an unpacked wheel with incompatible
          Wheel-Version
        * when the .dist-info dir does not match the wheel
    """
    info_dir, metadata = parse_wheel(wheel_zip, name)
    if wheel_root_is_purelib(metadata):
        lib_dir = scheme.purelib
    else:
        lib_dir = scheme.platlib
    installed: Dict[RecordPath, RecordPath] = {}
    changed: Set[RecordPath] = set()
    generated: List[str] = []

    def record_installed(srcfile: RecordPath, destfile: str, modified: bool=False) -> None:
        """Map archive RECORD paths to installation RECORD paths."""
        newpath = _fs_to_record_path(destfile, lib_dir)
        installed[srcfile] = newpath
        if modified:
            changed.add(newpath)

    def is_dir_path(path: RecordPath) -> bool:
        return path.endswith('/')

    def assert_no_path_traversal(dest_dir_path: str, target_path: str) -> None:
        if not is_within_directory(dest_dir_path, target_path):
            message = 'The wheel {!r} has a file {!r} trying to install outside the target directory {!r}'
            raise InstallationError(message.format(wheel_path, target_path, dest_dir_path))

    def root_scheme_file_maker(zip_file: ZipFile, dest: str) -> Callable[[RecordPath], 'File']:

        def make_root_scheme_file(record_path: RecordPath) -> 'File':
            normed_path = os.path.normpath(record_path)
            dest_path = os.path.join(dest, normed_path)
            assert_no_path_traversal(dest, dest_path)
            return ZipBackedFile(record_path, dest_path, zip_file)
        return make_root_scheme_file

    def data_scheme_file_maker(zip_file: ZipFile, scheme: Scheme) -> Callable[[RecordPath], 'File']:
        scheme_paths = {key: getattr(scheme, key) for key in SCHEME_KEYS}

        def make_data_scheme_file(record_path: RecordPath) -> 'File':
            normed_path = os.path.normpath(record_path)
            try:
                _, scheme_key, dest_subpath = normed_path.split(os.path.sep, 2)
            except ValueError:
                message = "Unexpected file in {}: {!r}. .data directory contents should be named like: '<scheme key>/<path>'.".format(wheel_path, record_path)
                raise InstallationError(message)
            try:
                scheme_path = scheme_paths[scheme_key]
            except KeyError:
                valid_scheme_keys = ', '.join(sorted(scheme_paths))
                message = 'Unknown scheme key used in {}: {} (for file {!r}). .data directory contents should be in subdirectories named with a valid scheme key ({})'.format(wheel_path, scheme_key, record_path, valid_scheme_keys)
                raise InstallationError(message)
            dest_path = os.path.join(scheme_path, dest_subpath)
            assert_no_path_traversal(scheme_path, dest_path)
            return ZipBackedFile(record_path, dest_path, zip_file)
        return make_data_scheme_file

    def is_data_scheme_path(path: RecordPath) -> bool:
        return path.split('/', 1)[0].endswith('.data')
    paths = cast(List[RecordPath], wheel_zip.namelist())
    file_paths = filterfalse(is_dir_path, paths)
    root_scheme_paths, data_scheme_paths = partition(is_data_scheme_path, file_paths)
    make_root_scheme_file = root_scheme_file_maker(wheel_zip, lib_dir)
    files: Iterator[File] = map(make_root_scheme_file, root_scheme_paths)

    def is_script_scheme_path(path: RecordPath) -> bool:
        parts = path.split('/', 2)
        return len(parts) > 2 and parts[0].endswith('.data') and (parts[1] == 'scripts')
    other_scheme_paths, script_scheme_paths = partition(is_script_scheme_path, data_scheme_paths)
    make_data_scheme_file = data_scheme_file_maker(wheel_zip, scheme)
    other_scheme_files = map(make_data_scheme_file, other_scheme_paths)
    files = chain(files, other_scheme_files)
    distribution = get_wheel_distribution(FilesystemWheel(wheel_path), canonicalize_name(name))
    console, gui = get_entrypoints(distribution)

    def is_entrypoint_wrapper(file: 'File') -> bool:
        path = file.dest_path
        name = os.path.basename(path)
        if name.lower().endswith('.exe'):
            matchname = name[:-4]
        elif name.lower().endswith('-script.py'):
            matchname = name[:-10]
        elif name.lower().endswith('.pya'):
            matchname = name[:-4]
        else:
            matchname = name
        return matchname in console or matchname in gui
    script_scheme_files: Iterator[File] = map(make_data_scheme_file, script_scheme_paths)
    script_scheme_files = filterfalse(is_entrypoint_wrapper, script_scheme_files)
    script_scheme_files = map(ScriptFile, script_scheme_files)
    files = chain(files, script_scheme_files)
    for file in files:
        file.save()
        record_installed(file.src_record_path, file.dest_path, file.changed)

    def pyc_source_file_paths() -> Generator[str, None, None]:
        for installed_path in sorted(set(installed.values())):
            full_installed_path = os.path.join(lib_dir, installed_path)
            if not os.path.isfile(full_installed_path):
                continue
            if not full_installed_path.endswith('.py'):
                continue
            yield full_installed_path

    def pyc_output_path(path: str) -> str:
        """Return the path the pyc file would have been written to."""
        return importlib.util.cache_from_source(path)
    if pycompile:
        with captured_stdout() as stdout:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                for path in pyc_source_file_paths():
                    success = compileall.compile_file(path, force=True, quiet=True)
                    if success:
                        pyc_path = pyc_output_path(path)
                        assert os.path.exists(pyc_path)
                        pyc_record_path = cast('RecordPath', pyc_path.replace(os.path.sep, '/'))
                        record_installed(pyc_record_path, pyc_path)
        logger.debug(stdout.getvalue())
    maker = PipScriptMaker(None, scheme.scripts)
    maker.clobber = True
    maker.variants = {''}
    maker.set_mode = True
    scripts_to_generate = get_console_script_specs(console)
    gui_scripts_to_generate = list(starmap('{} = {}'.format, gui.items()))
    generated_console_scripts = maker.make_multiple(scripts_to_generate)
    generated.extend(generated_console_scripts)
    generated.extend(maker.make_multiple(gui_scripts_to_generate, {'gui': True}))
    if warn_script_location:
        msg = message_about_scripts_not_on_PATH(generated_console_scripts)
        if msg is not None:
            logger.warning(msg)
    generated_file_mode = 438 & ~current_umask()

    @contextlib.contextmanager
    def _generate_file(path: str, **kwargs: Any) -> Generator[BinaryIO, None, None]:
        with adjacent_tmp_file(path, **kwargs) as f:
            yield f
        os.chmod(f.name, generated_file_mode)
        replace(f.name, path)
    dest_info_dir = os.path.join(lib_dir, info_dir)
    installer_path = os.path.join(dest_info_dir, 'INSTALLER')
    with _generate_file(installer_path) as installer_file:
        installer_file.write(b'pip\n')
    generated.append(installer_path)
    if direct_url is not None:
        direct_url_path = os.path.join(dest_info_dir, DIRECT_URL_METADATA_NAME)
        with _generate_file(direct_url_path) as direct_url_file:
            direct_url_file.write(direct_url.to_json().encode('utf-8'))
        generated.append(direct_url_path)
    if requested:
        requested_path = os.path.join(dest_info_dir, 'REQUESTED')
        with open(requested_path, 'wb'):
            pass
        generated.append(requested_path)
    record_text = distribution.read_text('RECORD')
    record_rows = list(csv.reader(record_text.splitlines()))
    rows = get_csv_rows_for_installed(record_rows, installed=installed, changed=changed, generated=generated, lib_dir=lib_dir)
    record_path = os.path.join(dest_info_dir, 'RECORD')
    with _generate_file(record_path, **csv_io_kwargs('w')) as record_file:
        writer = csv.writer(cast('IO[str]', record_file))
        writer.writerows(_normalized_outrows(rows))