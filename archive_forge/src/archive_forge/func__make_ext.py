import importlib.metadata as importlib_metadata
from stevedore import extension
from stevedore import sphinxext
from stevedore.tests import utils
def _make_ext(name, docstring):

    def inner():
        pass
    inner.__doc__ = docstring
    m1 = importlib_metadata.EntryPoint(name, '{}_module:{}'.format(name, name), 'group')
    return extension.Extension(name, m1, inner, None)