from typing import List, Optional, Tuple, Union
from transformers import (AutoTokenizer, PreTrainedTokenizer,
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.utils import make_async, LRUCache
from vllm.transformers_utils.tokenizers import *
def _convert_tokens_to_string_with_added_encoders(tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], output_tokens: List[str], skip_special_tokens: bool, spaces_between_special_tokens: bool) -> str:
    sub_texts = []
    current_sub_text = []
    all_special_tokens = set(tokenizer.all_special_tokens)
    for token in output_tokens:
        if skip_special_tokens and token in all_special_tokens:
            continue
        if token in tokenizer.get_added_vocab():
            if current_sub_text:
                sub_text = tokenizer.convert_tokens_to_string(current_sub_text)
                sub_texts.append(sub_text)
                current_sub_text = []
            sub_texts.append(token)
        else:
            current_sub_text.append(token)
    if current_sub_text:
        sub_text = tokenizer.convert_tokens_to_string(current_sub_text)
        sub_texts.append(sub_text)
    if spaces_between_special_tokens:
        return ' '.join(sub_texts)
    else:
        return ''.join(sub_texts)