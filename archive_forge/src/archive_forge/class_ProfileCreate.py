import os
from traitlets.config.application import Application
from IPython.core.application import (
from IPython.core.profiledir import ProfileDir
from IPython.utils.importstring import import_item
from IPython.paths import get_ipython_dir, get_ipython_package_dir
from traitlets import Unicode, Bool, Dict, observe
class ProfileCreate(BaseIPythonApplication):
    name = u'ipython-profile'
    description = create_help
    examples = _create_examples
    auto_create = Bool(True).tag(config=True)

    def _log_format_default(self):
        return '[%(name)s] %(message)s'

    def _copy_config_files_default(self):
        return True
    parallel = Bool(False, help='whether to include parallel computing config files').tag(config=True)

    @observe('parallel')
    def _parallel_changed(self, change):
        parallel_files = ['ipcontroller_config.py', 'ipengine_config.py', 'ipcluster_config.py']
        if change['new']:
            for cf in parallel_files:
                self.config_files.append(cf)
        else:
            for cf in parallel_files:
                if cf in self.config_files:
                    self.config_files.remove(cf)

    def parse_command_line(self, argv):
        super(ProfileCreate, self).parse_command_line(argv)
        if self.extra_args:
            self.profile = self.extra_args[0]
    flags = Dict(create_flags)
    classes = [ProfileDir]

    def _import_app(self, app_path):
        """import an app class"""
        app = None
        name = app_path.rsplit('.', 1)[-1]
        try:
            app = import_item(app_path)
        except ImportError:
            self.log.info("Couldn't import %s, config file will be excluded", name)
        except Exception:
            self.log.warning('Unexpected error importing %s', name, exc_info=True)
        return app

    def init_config_files(self):
        super(ProfileCreate, self).init_config_files()
        from IPython.terminal.ipapp import TerminalIPythonApp
        apps = [TerminalIPythonApp]
        for app_path in ('ipykernel.kernelapp.IPKernelApp',):
            app = self._import_app(app_path)
            if app is not None:
                apps.append(app)
        if self.parallel:
            from ipyparallel.apps.ipcontrollerapp import IPControllerApp
            from ipyparallel.apps.ipengineapp import IPEngineApp
            from ipyparallel.apps.ipclusterapp import IPClusterStart
            apps.extend([IPControllerApp, IPEngineApp, IPClusterStart])
        for App in apps:
            app = App()
            app.config.update(self.config)
            app.log = self.log
            app.overwrite = self.overwrite
            app.copy_config_files = True
            app.ipython_dir = self.ipython_dir
            app.profile_dir = self.profile_dir
            app.init_config_files()

    def stage_default_config_file(self):
        pass