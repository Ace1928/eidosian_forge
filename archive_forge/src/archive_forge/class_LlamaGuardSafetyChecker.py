import os
import torch
import warnings
from typing import List
from string import Template
from enum import Enum
class LlamaGuardSafetyChecker(object):

    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from llama_recipes.inference.prompt_format_utils import build_default_prompt, create_conversation, LlamaGuardVersion
        model_id = 'meta-llama/LlamaGuard-7b'
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map='auto')

    def __call__(self, output_text, **kwargs):
        agent_type = kwargs.get('agent_type', AgentType.USER)
        user_prompt = kwargs.get('user_prompt', '')
        model_prompt = output_text.strip()
        if agent_type == AgentType.AGENT:
            if user_prompt == '':
                print('empty user prompt for agent check, returning unsafe')
                return ('Llama Guard', False, 'Missing user_prompt from Agent response check')
            else:
                model_prompt = model_prompt.replace(user_prompt, '')
                user_prompt = f'User: {user_prompt}'
                agent_prompt = f'Agent: {model_prompt}'
                chat = [{'role': 'user', 'content': user_prompt}, {'role': 'assistant', 'content': agent_prompt}]
        else:
            chat = [{'role': 'user', 'content': model_prompt}]
        input_ids = self.tokenizer.apply_chat_template(chat, return_tensors='pt').to('cuda')
        prompt_len = input_ids.shape[-1]
        output = self.model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
        result = self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
        splitted_result = result.split('\n')[0]
        is_safe = splitted_result == 'safe'
        report = result
        return ('Llama Guard', is_safe, report)