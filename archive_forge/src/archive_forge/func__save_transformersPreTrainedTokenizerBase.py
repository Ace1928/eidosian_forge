import os
import sys
from io import BytesIO
from types import CodeType, FunctionType
import dill
from packaging import version
from .. import config
def _save_transformersPreTrainedTokenizerBase(pickler, obj):
    log(pickler, f'Tok: {obj}')
    state = obj.__dict__
    if 'cache' in state and isinstance(state['cache'], dict):
        state['cache'] = {}
    pickler.save_reduce(type(obj), (), state=state, obj=obj)
    log(pickler, '# Tok')