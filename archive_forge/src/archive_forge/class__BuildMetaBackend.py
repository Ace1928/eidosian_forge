import io
import os
import shlex
import sys
import tokenize
import shutil
import contextlib
import tempfile
import warnings
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union
import setuptools
import distutils
from . import errors
from ._path import same_path
from ._reqs import parse_strings
from .warnings import SetuptoolsDeprecationWarning
from distutils.util import strtobool
class _BuildMetaBackend(_ConfigSettingsTranslator):

    def _get_build_requires(self, config_settings, requirements):
        sys.argv = [*sys.argv[:1], *self._global_args(config_settings), 'egg_info']
        try:
            with Distribution.patch():
                self.run_setup()
        except SetupRequirementsError as e:
            requirements += e.specifiers
        return requirements

    def run_setup(self, setup_script='setup.py'):
        __file__ = os.path.abspath(setup_script)
        __name__ = '__main__'
        with _open_setup_script(__file__) as f:
            code = f.read().replace('\\r\\n', '\\n')
        try:
            exec(code, locals())
        except SystemExit as e:
            if e.code:
                raise
            SetuptoolsDeprecationWarning.emit('Running `setup.py` directly as CLI tool is deprecated.', "Please avoid using `sys.exit(0)` or similar statements that don't fit in the paradigm of a configuration file.", see_url='https://blog.ganssle.io/articles/2021/10/setup-py-deprecated.html')

    def get_requires_for_build_wheel(self, config_settings=None):
        return self._get_build_requires(config_settings, requirements=['wheel'])

    def get_requires_for_build_sdist(self, config_settings=None):
        return self._get_build_requires(config_settings, requirements=[])

    def _bubble_up_info_directory(self, metadata_directory: str, suffix: str) -> str:
        """
        PEP 517 requires that the .dist-info directory be placed in the
        metadata_directory. To comply, we MUST copy the directory to the root.

        Returns the basename of the info directory, e.g. `proj-0.0.0.dist-info`.
        """
        info_dir = self._find_info_directory(metadata_directory, suffix)
        if not same_path(info_dir.parent, metadata_directory):
            shutil.move(str(info_dir), metadata_directory)
        return info_dir.name

    def _find_info_directory(self, metadata_directory: str, suffix: str) -> Path:
        for parent, dirs, _ in os.walk(metadata_directory):
            candidates = [f for f in dirs if f.endswith(suffix)]
            if len(candidates) != 0 or len(dirs) != 1:
                assert len(candidates) == 1, f'Multiple {suffix} directories found'
                return Path(parent, candidates[0])
        msg = f'No {suffix} directory found in {metadata_directory}'
        raise errors.InternalError(msg)

    def prepare_metadata_for_build_wheel(self, metadata_directory, config_settings=None):
        sys.argv = [*sys.argv[:1], *self._global_args(config_settings), 'dist_info', '--output-dir', metadata_directory, '--keep-egg-info']
        with no_install_setup_requires():
            self.run_setup()
        self._bubble_up_info_directory(metadata_directory, '.egg-info')
        return self._bubble_up_info_directory(metadata_directory, '.dist-info')

    def _build_with_temp_dir(self, setup_command, result_extension, result_directory, config_settings):
        result_directory = os.path.abspath(result_directory)
        os.makedirs(result_directory, exist_ok=True)
        temp_opts = {'prefix': '.tmp-', 'dir': result_directory}
        with tempfile.TemporaryDirectory(**temp_opts) as tmp_dist_dir:
            sys.argv = [*sys.argv[:1], *self._global_args(config_settings), *setup_command, '--dist-dir', tmp_dist_dir]
            with no_install_setup_requires():
                self.run_setup()
            result_basename = _file_with_extension(tmp_dist_dir, result_extension)
            result_path = os.path.join(result_directory, result_basename)
            if os.path.exists(result_path):
                os.remove(result_path)
            os.rename(os.path.join(tmp_dist_dir, result_basename), result_path)
        return result_basename

    def build_wheel(self, wheel_directory, config_settings=None, metadata_directory=None):
        with suppress_known_deprecation():
            return self._build_with_temp_dir(['bdist_wheel', *self._arbitrary_args(config_settings)], '.whl', wheel_directory, config_settings)

    def build_sdist(self, sdist_directory, config_settings=None):
        return self._build_with_temp_dir(['sdist', '--formats', 'gztar'], '.tar.gz', sdist_directory, config_settings)

    def _get_dist_info_dir(self, metadata_directory: Optional[str]) -> Optional[str]:
        if not metadata_directory:
            return None
        dist_info_candidates = list(Path(metadata_directory).glob('*.dist-info'))
        assert len(dist_info_candidates) <= 1
        return str(dist_info_candidates[0]) if dist_info_candidates else None
    if not LEGACY_EDITABLE:

        def build_editable(self, wheel_directory, config_settings=None, metadata_directory=None):
            info_dir = self._get_dist_info_dir(metadata_directory)
            opts = ['--dist-info-dir', info_dir] if info_dir else []
            cmd = ['editable_wheel', *opts, *self._editable_args(config_settings)]
            with suppress_known_deprecation():
                return self._build_with_temp_dir(cmd, '.whl', wheel_directory, config_settings)

        def get_requires_for_build_editable(self, config_settings=None):
            return self.get_requires_for_build_wheel(config_settings)

        def prepare_metadata_for_build_editable(self, metadata_directory, config_settings=None):
            return self.prepare_metadata_for_build_wheel(metadata_directory, config_settings)