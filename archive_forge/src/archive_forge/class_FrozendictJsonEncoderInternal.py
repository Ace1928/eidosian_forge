from .version import version as __version__
from . import monkeypatch
from .cool import *
from collections.abc import Mapping
class FrozendictJsonEncoderInternal(BaseJsonEncoder):

    def default(self, obj):
        if isinstance(obj, frozendict):
            return dict(obj)
        return BaseJsonEncoder.default(self, obj)