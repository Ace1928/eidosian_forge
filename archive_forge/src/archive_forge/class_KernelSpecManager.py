from __future__ import annotations
import json
import os
import re
import shutil
import typing as t
import warnings
from jupyter_core.paths import SYSTEM_JUPYTER_PATH, jupyter_data_dir, jupyter_path
from traitlets import Bool, CaselessStrEnum, Dict, HasTraits, List, Set, Type, Unicode, observe
from traitlets.config import LoggingConfigurable
from .provisioning import KernelProvisionerFactory as KPF  # noqa
class KernelSpecManager(LoggingConfigurable):
    """A manager for kernel specs."""
    kernel_spec_class = Type(KernelSpec, config=True, help='The kernel spec class.  This is configurable to allow\n        subclassing of the KernelSpecManager for customized behavior.\n        ')
    ensure_native_kernel = Bool(True, config=True, help='If there is no Python kernelspec registered and the IPython\n        kernel is available, ensure it is added to the spec list.\n        ')
    data_dir = Unicode()

    def _data_dir_default(self) -> str:
        return jupyter_data_dir()
    user_kernel_dir = Unicode()

    def _user_kernel_dir_default(self) -> str:
        return pjoin(self.data_dir, 'kernels')
    whitelist = Set(config=True, help='Deprecated, use `KernelSpecManager.allowed_kernelspecs`\n        ')
    allowed_kernelspecs = Set(config=True, help='List of allowed kernel names.\n\n        By default, all installed kernels are allowed.\n        ')
    kernel_dirs: List[str] = List(help='List of kernel directories to search. Later ones take priority over earlier.')
    _deprecated_aliases = {'whitelist': ('allowed_kernelspecs', '7.0')}

    @observe(*list(_deprecated_aliases))
    def _deprecated_trait(self, change: t.Any) -> None:
        """observer for deprecated traits"""
        old_attr = change.name
        new_attr, version = self._deprecated_aliases[old_attr]
        new_value = getattr(self, new_attr)
        if new_value != change.new:
            self.log.warning(f'{self.__class__.__name__}.{old_attr} is deprecated in jupyter_client {version}, use {self.__class__.__name__}.{new_attr} instead')
            setattr(self, new_attr, change.new)

    def _kernel_dirs_default(self) -> list[str]:
        dirs = jupyter_path('kernels')
        try:
            from IPython.paths import get_ipython_dir
            dirs.append(os.path.join(get_ipython_dir(), 'kernels'))
        except ModuleNotFoundError:
            pass
        return dirs

    def find_kernel_specs(self) -> dict[str, str]:
        """Returns a dict mapping kernel names to resource directories."""
        d = {}
        for kernel_dir in self.kernel_dirs:
            kernels = _list_kernels_in(kernel_dir)
            for kname, spec in kernels.items():
                if kname not in d:
                    self.log.debug('Found kernel %s in %s', kname, kernel_dir)
                    d[kname] = spec
        if self.ensure_native_kernel and NATIVE_KERNEL_NAME not in d:
            try:
                from ipykernel.kernelspec import RESOURCES
                self.log.debug('Native kernel (%s) available from %s', NATIVE_KERNEL_NAME, RESOURCES)
                d[NATIVE_KERNEL_NAME] = RESOURCES
            except ImportError:
                self.log.warning('Native kernel (%s) is not available', NATIVE_KERNEL_NAME)
        if self.allowed_kernelspecs:
            d = {name: spec for name, spec in d.items() if name in self.allowed_kernelspecs}
        return d

    def _get_kernel_spec_by_name(self, kernel_name: str, resource_dir: str) -> KernelSpec:
        """Returns a :class:`KernelSpec` instance for a given kernel_name
        and resource_dir.
        """
        kspec = None
        if kernel_name == NATIVE_KERNEL_NAME:
            try:
                from ipykernel.kernelspec import RESOURCES, get_kernel_dict
            except ImportError:
                pass
            else:
                if resource_dir == RESOURCES:
                    kdict = get_kernel_dict()
                    kspec = self.kernel_spec_class(resource_dir=resource_dir, **kdict)
        if not kspec:
            kspec = self.kernel_spec_class.from_resource_dir(resource_dir)
        if not KPF.instance(parent=self.parent).is_provisioner_available(kspec):
            raise NoSuchKernel(kernel_name)
        return kspec

    def _find_spec_directory(self, kernel_name: str) -> str | None:
        """Find the resource directory of a named kernel spec"""
        for kernel_dir in [kd for kd in self.kernel_dirs if os.path.isdir(kd)]:
            files = os.listdir(kernel_dir)
            for f in files:
                path = pjoin(kernel_dir, f)
                if f.lower() == kernel_name and _is_kernel_dir(path):
                    return path
        if kernel_name == NATIVE_KERNEL_NAME:
            try:
                from ipykernel.kernelspec import RESOURCES
            except ImportError:
                pass
            else:
                return RESOURCES
        return None

    def get_kernel_spec(self, kernel_name: str) -> KernelSpec:
        """Returns a :class:`KernelSpec` instance for the given kernel_name.

        Raises :exc:`NoSuchKernel` if the given kernel name is not found.
        """
        if not _is_valid_kernel_name(kernel_name):
            self.log.warning(f'Kernelspec name {kernel_name} is invalid: {_kernel_name_description}')
        resource_dir = self._find_spec_directory(kernel_name.lower())
        if resource_dir is None:
            self.log.warning('Kernelspec name %s cannot be found!', kernel_name)
            raise NoSuchKernel(kernel_name)
        return self._get_kernel_spec_by_name(kernel_name, resource_dir)

    def get_all_specs(self) -> dict[str, t.Any]:
        """Returns a dict mapping kernel names to kernelspecs.

        Returns a dict of the form::

            {
              'kernel_name': {
                'resource_dir': '/path/to/kernel_name',
                'spec': {"the spec itself": ...}
              },
              ...
            }
        """
        d = self.find_kernel_specs()
        res = {}
        for kname, resource_dir in d.items():
            try:
                if self.__class__ is KernelSpecManager:
                    spec = self._get_kernel_spec_by_name(kname, resource_dir)
                else:
                    spec = self.get_kernel_spec(kname)
                res[kname] = {'resource_dir': resource_dir, 'spec': spec.to_dict()}
            except NoSuchKernel:
                pass
            except Exception:
                self.log.warning('Error loading kernelspec %r', kname, exc_info=True)
        return res

    def remove_kernel_spec(self, name: str) -> str:
        """Remove a kernel spec directory by name.

        Returns the path that was deleted.
        """
        save_native = self.ensure_native_kernel
        try:
            self.ensure_native_kernel = False
            specs = self.find_kernel_specs()
        finally:
            self.ensure_native_kernel = save_native
        spec_dir = specs[name]
        self.log.debug('Removing %s', spec_dir)
        if os.path.islink(spec_dir):
            os.remove(spec_dir)
        else:
            shutil.rmtree(spec_dir)
        return spec_dir

    def _get_destination_dir(self, kernel_name: str, user: bool=False, prefix: str | None=None) -> str:
        if user:
            return os.path.join(self.user_kernel_dir, kernel_name)
        elif prefix:
            return os.path.join(os.path.abspath(prefix), 'share', 'jupyter', 'kernels', kernel_name)
        else:
            return os.path.join(SYSTEM_JUPYTER_PATH[0], 'kernels', kernel_name)

    def install_kernel_spec(self, source_dir: str, kernel_name: str | None=None, user: bool=False, replace: bool | None=None, prefix: str | None=None) -> str:
        """Install a kernel spec by copying its directory.

        If ``kernel_name`` is not given, the basename of ``source_dir`` will
        be used.

        If ``user`` is False, it will attempt to install into the systemwide
        kernel registry. If the process does not have appropriate permissions,
        an :exc:`OSError` will be raised.

        If ``prefix`` is given, the kernelspec will be installed to
        PREFIX/share/jupyter/kernels/KERNEL_NAME. This can be sys.prefix
        for installation inside virtual or conda envs.
        """
        source_dir = source_dir.rstrip('/\\')
        if not kernel_name:
            kernel_name = os.path.basename(source_dir)
        kernel_name = kernel_name.lower()
        if not _is_valid_kernel_name(kernel_name):
            msg = f'Invalid kernel name {kernel_name!r}.  {_kernel_name_description}'
            raise ValueError(msg)
        if user and prefix:
            msg = "Can't specify both user and prefix. Please choose one or the other."
            raise ValueError(msg)
        if replace is not None:
            warnings.warn('replace is ignored. Installing a kernelspec always replaces an existing installation', DeprecationWarning, stacklevel=2)
        destination = self._get_destination_dir(kernel_name, user=user, prefix=prefix)
        self.log.debug('Installing kernelspec in %s', destination)
        kernel_dir = os.path.dirname(destination)
        if kernel_dir not in self.kernel_dirs:
            self.log.warning('Installing to %s, which is not in %s. The kernelspec may not be found.', kernel_dir, self.kernel_dirs)
        if os.path.isdir(destination):
            self.log.info('Removing existing kernelspec in %s', destination)
            shutil.rmtree(destination)
        shutil.copytree(source_dir, destination)
        self.log.info('Installed kernelspec %s in %s', kernel_name, destination)
        return destination

    def install_native_kernel_spec(self, user: bool=False) -> None:
        """DEPRECATED: Use ipykernel.kernelspec.install"""
        warnings.warn('install_native_kernel_spec is deprecated. Use ipykernel.kernelspec import install.', stacklevel=2)
        from ipykernel.kernelspec import install
        install(self, user=user)