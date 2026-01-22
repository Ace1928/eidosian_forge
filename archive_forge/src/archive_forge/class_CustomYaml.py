from typing import Union, IO, Any
from io import StringIO
import sys
from .ruamel_yaml import YAML
from .ruamel_yaml.representer import RepresenterError
from .util import force_path, FilePath, YAMLInput, YAMLOutput
class CustomYaml(YAML):

    def __init__(self, typ='safe', pure=True):
        YAML.__init__(self, typ=typ, pure=pure)
        self.default_flow_style = False
        self.allow_unicode = True
        self.encoding = 'utf-8'

    def dump(self, data, stream=None, **kw):
        inefficient = False
        if stream is None:
            inefficient = True
            stream = StringIO()
        YAML.dump(self, data, stream, **kw)
        if inefficient:
            return stream.getvalue()