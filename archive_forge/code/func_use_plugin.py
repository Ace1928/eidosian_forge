import os.path
import warnings
from configparser import ConfigParser
from glob import glob
from .collection import imread_collection_wrapper
def use_plugin(name, kind=None):
    """Set the default plugin for a specified operation.  The plugin
    will be loaded if it hasn't been already.

    Parameters
    ----------
    name : str
        Name of plugin. See ``skimage.io.available_plugins`` for a list of available
        plugins.
    kind : {'imsave', 'imread', 'imshow', 'imread_collection', 'imshow_collection'}, optional
        Set the plugin for this function.  By default,
        the plugin is set for all functions.

    Examples
    --------
    To use Matplotlib as the default image reader, you would write:

    >>> from skimage import io
    >>> io.use_plugin('matplotlib', 'imread')

    To see a list of available plugins run ``skimage.io.available_plugins``. Note
    that this lists plugins that are defined, but the full list may not be usable
    if your system does not have the required libraries installed.

    """
    if kind is None:
        kind = plugin_store.keys()
    else:
        if kind not in plugin_provides[name]:
            raise RuntimeError(f'Plugin {name} does not support `{kind}`.')
        if kind == 'imshow':
            kind = [kind, '_app_show']
        else:
            kind = [kind]
    _load(name)
    for k in kind:
        if k not in plugin_store:
            raise RuntimeError(f"'{k}' is not a known plugin function.")
        funcs = plugin_store[k]
        funcs = [(n, f) for n, f in funcs if n == name] + [(n, f) for n, f in funcs if n != name]
        plugin_store[k] = funcs