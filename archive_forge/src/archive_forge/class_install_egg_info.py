from distutils.cmd import Command
from distutils import log, dir_util
import os, sys, re
class install_egg_info(Command):
    """Install an .egg-info file for the package"""
    description = "Install package's PKG-INFO metadata as an .egg-info file"
    user_options = [('install-dir=', 'd', 'directory to install to')]

    def initialize_options(self):
        self.install_dir = None

    def finalize_options(self):
        self.set_undefined_options('install_lib', ('install_dir', 'install_dir'))
        basename = '%s-%s-py%d.%d.egg-info' % (to_filename(safe_name(self.distribution.get_name())), to_filename(safe_version(self.distribution.get_version())), *sys.version_info[:2])
        self.target = os.path.join(self.install_dir, basename)
        self.outputs = [self.target]

    def run(self):
        target = self.target
        if os.path.isdir(target) and (not os.path.islink(target)):
            dir_util.remove_tree(target, dry_run=self.dry_run)
        elif os.path.exists(target):
            self.execute(os.unlink, (self.target,), 'Removing ' + target)
        elif not os.path.isdir(self.install_dir):
            self.execute(os.makedirs, (self.install_dir,), 'Creating ' + self.install_dir)
        log.info('Writing %s', target)
        if not self.dry_run:
            with open(target, 'w', encoding='UTF-8') as f:
                self.distribution.metadata.write_pkg_file(f)

    def get_outputs(self):
        return self.outputs