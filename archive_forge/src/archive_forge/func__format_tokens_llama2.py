import sys
from pathlib import Path
from typing import List, Literal, TypedDict
from unittest.mock import patch
import pytest
import torch
from llama_recipes.inference.chat_utils import read_dialogs_from_file
def _format_tokens_llama2(dialogs, tokenizer):
    prompt_tokens = []
    for dialog in dialogs:
        if dialog[0]['role'] == 'system':
            dialog = [{'role': dialog[1]['role'], 'content': B_SYS + dialog[0]['content'] + E_SYS + dialog[1]['content']}] + dialog[2:]
        assert all([msg['role'] == 'user' for msg in dialog[::2]]) and all([msg['role'] == 'assistant' for msg in dialog[1::2]]), "model only supports 'system','user' and 'assistant' roles, starting with user and alternating (u/a/u/a/u...)"
        '\n        Please verify that your tokenizer support adding "[INST]", "[/INST]" to your inputs.\n        Here, we are adding it manually.\n        '
        dialog_tokens: List[int] = sum([tokenizer.encode(f'{B_INST} {prompt['content'].strip()} {E_INST} {answer['content'].strip()} ') + [tokenizer.eos_token_id] for prompt, answer in zip(dialog[::2], dialog[1::2])], [])
        assert dialog[-1]['role'] == 'user', f'Last message must be from user, got {dialog[-1]['role']}'
        dialog_tokens += tokenizer.encode(f'{B_INST} {dialog[-1]['content'].strip()} {E_INST}')
        prompt_tokens.append(dialog_tokens)
    return prompt_tokens