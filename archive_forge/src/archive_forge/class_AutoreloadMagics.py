imported with ``from foo import ...`` was also updated.
from IPython.core import magic_arguments
from IPython.core.magic import Magics, magics_class, line_magic
import os
import sys
import traceback
import types
import weakref
import gc
import logging
from importlib import import_module, reload
from importlib.util import source_from_cache
@magics_class
class AutoreloadMagics(Magics):

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._reloader = ModuleReloader(self.shell)
        self._reloader.check_all = False
        self._reloader.autoload_obj = False
        self.loaded_modules = set(sys.modules)

    @line_magic
    @magic_arguments.magic_arguments()
    @magic_arguments.argument('mode', type=str, default='now', nargs='?', help="blank or 'now' - Reload all modules (except those excluded by %%aimport)\n             automatically now.\n\n             '0' or 'off' - Disable automatic reloading.\n\n             '1' or 'explicit' - Reload only modules imported with %%aimport every\n             time before executing the Python code typed.\n\n             '2' or 'all' - Reload all modules (except those excluded by %%aimport)\n             every time before executing the Python code typed.\n\n             '3' or 'complete' - Same as 2/all, but also but also adds any new\n             objects in the module.\n             ")
    @magic_arguments.argument('-p', '--print', action='store_true', default=False, help='Show autoreload activity using `print` statements')
    @magic_arguments.argument('-l', '--log', action='store_true', default=False, help='Show autoreload activity using the logger')
    @magic_arguments.argument('--hide-errors', action='store_true', default=False, help='Hide autoreload errors')
    def autoreload(self, line=''):
        """%autoreload => Reload modules automatically

        %autoreload or %autoreload now
        Reload all modules (except those excluded by %aimport) automatically
        now.

        %autoreload 0 or %autoreload off
        Disable automatic reloading.

        %autoreload 1 or %autoreload explicit
        Reload only modules imported with %aimport every time before executing
        the Python code typed.

        %autoreload 2 or %autoreload all
        Reload all modules (except those excluded by %aimport) every time
        before executing the Python code typed.

        %autoreload 3 or %autoreload complete
        Same as 2/all, but also but also adds any new objects in the module. See
        unit test at IPython/extensions/tests/test_autoreload.py::test_autoload_newly_added_objects

        The optional arguments --print and --log control display of autoreload activity. The default
        is to act silently; --print (or -p) will print out the names of modules that are being
        reloaded, and --log (or -l) outputs them to the log at INFO level.

        The optional argument --hide-errors hides any errors that can happen when trying to
        reload code.

        Reloading Python modules in a reliable way is in general
        difficult, and unexpected things may occur. %autoreload tries to
        work around common pitfalls by replacing function code objects and
        parts of classes previously in the module with new versions. This
        makes the following things to work:

        - Functions and classes imported via 'from xxx import foo' are upgraded
          to new versions when 'xxx' is reloaded.

        - Methods and properties of classes are upgraded on reload, so that
          calling 'c.foo()' on an object 'c' created before the reload causes
          the new code for 'foo' to be executed.

        Some of the known remaining caveats are:

        - Replacing code objects does not always succeed: changing a @property
          in a class to an ordinary method or a method to a member variable
          can cause problems (but in old objects only).

        - Functions that are removed (eg. via monkey-patching) from a module
          before it is reloaded are not upgraded.

        - C extension modules cannot be reloaded, and so cannot be
          autoreloaded.

        """
        args = magic_arguments.parse_argstring(self.autoreload, line)
        mode = args.mode.lower()
        p = print
        logger = logging.getLogger('autoreload')
        l = logger.info

        def pl(msg):
            p(msg)
            l(msg)
        if args.print is False and args.log is False:
            self._reloader._report = lambda msg: None
        elif args.print is True:
            if args.log is True:
                self._reloader._report = pl
            else:
                self._reloader._report = p
        elif args.log is True:
            self._reloader._report = l
        self._reloader.hide_errors = args.hide_errors
        if mode == '' or mode == 'now':
            self._reloader.check(True)
        elif mode == '0' or mode == 'off':
            self._reloader.enabled = False
        elif mode == '1' or mode == 'explicit':
            self._reloader.enabled = True
            self._reloader.check_all = False
            self._reloader.autoload_obj = False
        elif mode == '2' or mode == 'all':
            self._reloader.enabled = True
            self._reloader.check_all = True
            self._reloader.autoload_obj = False
        elif mode == '3' or mode == 'complete':
            self._reloader.enabled = True
            self._reloader.check_all = True
            self._reloader.autoload_obj = True
        else:
            raise ValueError(f'Unrecognized autoreload mode "{mode}".')

    @line_magic
    def aimport(self, parameter_s='', stream=None):
        """%aimport => Import modules for automatic reloading.

        %aimport
        List modules to automatically import and not to import.

        %aimport foo
        Import module 'foo' and mark it to be autoreloaded for %autoreload explicit

        %aimport foo, bar
        Import modules 'foo', 'bar' and mark them to be autoreloaded for %autoreload explicit

        %aimport -foo, bar
        Mark module 'foo' to not be autoreloaded for %autoreload explicit, all, or complete, and 'bar'
        to be autoreloaded for mode explicit.
        """
        modname = parameter_s
        if not modname:
            to_reload = sorted(self._reloader.modules.keys())
            to_skip = sorted(self._reloader.skip_modules.keys())
            if stream is None:
                stream = sys.stdout
            if self._reloader.check_all:
                stream.write('Modules to reload:\nall-except-skipped\n')
            else:
                stream.write('Modules to reload:\n%s\n' % ' '.join(to_reload))
            stream.write('\nModules to skip:\n%s\n' % ' '.join(to_skip))
        else:
            for _module in [_.strip() for _ in modname.split(',')]:
                if _module.startswith('-'):
                    _module = _module[1:].strip()
                    self._reloader.mark_module_skipped(_module)
                else:
                    top_module, top_name = self._reloader.aimport_module(_module)
                    self.shell.push({top_name: top_module})

    def pre_run_cell(self, info):
        if self._reloader.enabled:
            try:
                self._reloader.check()
            except:
                pass

    def post_execute_hook(self):
        """Cache the modification times of any modules imported in this execution"""
        newly_loaded_modules = set(sys.modules) - self.loaded_modules
        for modname in newly_loaded_modules:
            _, pymtime = self._reloader.filename_and_mtime(sys.modules[modname])
            if pymtime is not None:
                self._reloader.modules_mtimes[modname] = pymtime
        self.loaded_modules.update(newly_loaded_modules)