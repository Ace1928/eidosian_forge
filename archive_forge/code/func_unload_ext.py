from IPython.core.error import UsageError
from IPython.core.magic import Magics, magics_class, line_magic
@line_magic
def unload_ext(self, module_str):
    """Unload an IPython extension by its module name.

        Not all extensions can be unloaded, only those which define an
        ``unload_ipython_extension`` function.
        """
    if not module_str:
        raise UsageError('Missing module name.')
    res = self.shell.extension_manager.unload_extension(module_str)
    if res == 'no unload function':
        print("The %s extension doesn't define how to unload it." % module_str)
    elif res == 'not loaded':
        print('The %s extension is not loaded.' % module_str)