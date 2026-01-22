import collections
import re
import sys
from typing import Any, Optional, Tuple, Type, Union
from absl import app
from pyglib import stringutil
class ModelReference(Reference):
    _required_fields = frozenset(('projectId', 'datasetId', 'modelId'))
    _format_str = '%(projectId)s:%(datasetId)s.%(modelId)s'
    typename = 'model'

    def __init__(self, **kwds):
        self.projectId: str = kwds['projectId']
        self.datasetId: str = kwds['datasetId']
        self.modelId: str = kwds['modelId']
        super().__init__(**kwds)