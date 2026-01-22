from distutils import cmd
import distutils.errors
import logging
import os
import sys
import warnings
class TestrReal(cmd.Command):
    description = 'DEPRECATED: Run unit tests using testr'
    user_options = [('coverage', None, 'Replace PYTHON with coverage and merge coverage from each testr worker.'), ('testr-args=', 't', "Run 'testr' with these args"), ('omit=', 'o', 'Files to omit from coverage calculations'), ('coverage-package-name=', None, 'Use this name to select packages for coverage (one or more, comma-separated)'), ('slowest', None, 'Show slowest test times after tests complete.'), ('no-parallel', None, 'Run testr serially'), ('log-level=', 'l', 'Log level (default: info)')]
    boolean_options = ['coverage', 'slowest', 'no_parallel']

    def _run_testr(self, *args):
        logger.debug('_run_testr called with args = %r', args)
        return commands.run_argv([sys.argv[0]] + list(args), sys.stdin, sys.stdout, sys.stderr)

    def initialize_options(self):
        self.testr_args = None
        self.coverage = None
        self.omit = ''
        self.slowest = None
        self.coverage_package_name = None
        self.no_parallel = None
        self.log_level = 'info'

    def finalize_options(self):
        self.log_level = getattr(logging, self.log_level.upper(), logging.INFO)
        logging.basicConfig(level=self.log_level)
        logger.debug('finalize_options called')
        if self.testr_args is None:
            self.testr_args = []
        else:
            self.testr_args = self.testr_args.split()
        if self.omit:
            self.omit = '--omit=%s' % self.omit
        logger.debug('finalize_options: self.__dict__ = %r', self.__dict__)

    def run(self):
        """Set up testr repo, then run testr."""
        logger.debug('run called')
        warnings.warn("testr integration in pbr is deprecated. Please use the 'testr' setup command or call testr directly", DeprecationWarning)
        if not os.path.isdir('.testrepository'):
            self._run_testr('init')
        if self.coverage:
            self._coverage_before()
        if not self.no_parallel:
            testr_ret = self._run_testr('run', '--parallel', *self.testr_args)
        else:
            testr_ret = self._run_testr('run', *self.testr_args)
        if testr_ret:
            raise distutils.errors.DistutilsError('testr failed (%d)' % testr_ret)
        if self.slowest:
            print('Slowest Tests')
            self._run_testr('slowest')
        if self.coverage:
            self._coverage_after()

    def _coverage_before(self):
        logger.debug('_coverage_before called')
        package = self.distribution.get_name()
        if package.startswith('python-'):
            package = package[7:]
        if self.coverage_package_name:
            package = self.coverage_package_name
        options = '--source %s --parallel-mode' % package
        os.environ['PYTHON'] = 'coverage run %s' % options
        logger.debug("os.environ['PYTHON'] = %r", os.environ['PYTHON'])

    def _coverage_after(self):
        logger.debug('_coverage_after called')
        os.system('coverage combine')
        os.system('coverage html -d ./cover %s' % self.omit)
        os.system('coverage xml -o ./cover/coverage.xml %s' % self.omit)