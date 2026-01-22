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
def __preload(self):
    interface = self.interface
    user = self.user
    bot = self.bot
    preset_system = f"\nThe following is a coherent verbose detailed conversation between a girl named {bot} and her friend {user}. {bot} is very intelligent, creative and friendly. {bot} is unlikely to disagree with {user}, and {bot} doesn't like to ask {user} questions. {bot} likes to tell {user} a lot about herself and her opinions. {bot} usually gives {user} kind, helpful and informative advices.\n\n" if self.rwkv_type == RWKVType.Raven else f'{user}{interface} hi\n\n{bot}{interface} Hi. ' + 'I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.\n\n'
    logits, _ = self.run_rnn(self.fix_tokens(self.pipeline.encode(preset_system)))
    try:
        state_cache.add_state(state_cache.AddStateBody(prompt=preset_system, tokens=self.model_tokens, state=self.model_state, logits=logits))
    except HTTPException:
        pass