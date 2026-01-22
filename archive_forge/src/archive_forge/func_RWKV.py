from abc import ABC, abstractmethod
from enum import Enum, auto
import os
import pathlib
import copy
import re
from typing import Dict, Iterable, List, Tuple, Union, Type, Callable
from utils.log import quick_log
from fastapi import HTTPException
from pydantic import BaseModel, Field
from routes import state_cache
import global_var
def RWKV(model: str, strategy: str, tokenizer: Union[str, None]) -> AbstractRWKV:
    model = get_model_path(model)
    rwkv_beta = global_var.get(global_var.Args).rwkv_beta
    rwkv_cpp = getattr(global_var.get(global_var.Args), 'rwkv.cpp')
    webgpu = global_var.get(global_var.Args).webgpu
    if 'midi' in model.lower() or 'abc' in model.lower():
        os.environ['RWKV_RESCALE_LAYER'] = '999'
    if rwkv_beta:
        print('Using rwkv-beta')
        from rwkv_pip.beta.model import RWKV as Model
    elif rwkv_cpp:
        print('Using rwkv.cpp, strategy is ignored')
        from rwkv_pip.cpp.model import RWKV as Model
    elif webgpu:
        print('Using webgpu')
        from rwkv_pip.webgpu.model import RWKV as Model
    else:
        from rwkv_pip.model import RWKV as Model
    from rwkv_pip.utils import PIPELINE
    filename, _ = os.path.splitext(os.path.basename(model))
    model = Model(model, strategy)
    if not tokenizer:
        tokenizer = get_tokenizer(len(model.w['emb.weight']))
    pipeline = PIPELINE(model, tokenizer)
    rwkv_map: dict[str, Type[AbstractRWKV]] = {'20B_tokenizer': TextRWKV, 'rwkv_vocab_v20230424': TextRWKV, 'tokenizer-midi': MusicMidiRWKV, 'tokenizer-midipiano': MusicMidiRWKV, 'abc_tokenizer': MusicAbcRWKV}
    tokenizer_name = os.path.splitext(os.path.basename(tokenizer))[0]
    global_var.set(global_var.Midi_Vocab_Config_Type, global_var.MidiVocabConfig.Piano if tokenizer_name == 'tokenizer-midipiano' else global_var.MidiVocabConfig.Default)
    rwkv: AbstractRWKV
    if tokenizer_name in rwkv_map:
        rwkv = rwkv_map[tokenizer_name](model, pipeline)
    else:
        tokenizer_name = tokenizer_name.lower()
        if 'music' in tokenizer_name or 'midi' in tokenizer_name:
            rwkv = MusicMidiRWKV(model, pipeline)
        elif 'abc' in tokenizer_name:
            rwkv = MusicAbcRWKV(model, pipeline)
        else:
            rwkv = TextRWKV(model, pipeline)
    rwkv.name = filename
    rwkv.version = model.version
    return rwkv