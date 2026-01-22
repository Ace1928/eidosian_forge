from __future__ import absolute_import
import warnings
import textwrap
from ruamel.yaml.compat import utf8
class UnsafeLoaderWarning(YAMLWarning):
    text = "\nThe default 'Loader' for 'load(stream)' without further arguments can be unsafe.\nUse 'load(stream, Loader=ruamel.yaml.Loader)' explicitly if that is OK.\nAlternatively include the following in your code:\n\n  import warnings\n  warnings.simplefilter('ignore', ruamel.yaml.error.UnsafeLoaderWarning)\n\nIn most other cases you should consider using 'safe_load(stream)'"
    pass