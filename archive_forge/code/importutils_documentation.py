import sys
import traceback
Try to import a module from a list of modules.

    :param modules: A list of modules to try and import
    :returns: The first module found that can be imported
    :raises ImportError: If no modules can be imported from list

    .. versionadded:: 3.8
    