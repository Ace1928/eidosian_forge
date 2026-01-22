import copy
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional, Sequence
def _create_formatted_finetuning_example(training_example: TrainingExample, formatter_configs: FormatterConfigs, category_indices_to_include_in_llama_guard_prompt: List[int]) -> str:
    if formatter_configs.llama_guard_prompt_configs.should_shuffle_category_codes:
        random.shuffle(category_indices_to_include_in_llama_guard_prompt)
    else:
        category_indices_to_include_in_llama_guard_prompt = sorted(category_indices_to_include_in_llama_guard_prompt)
    llama_guard_prompt = _create_llama_guard_prompt(training_example, category_indices_to_include_in_llama_guard_prompt, formatter_configs)
    llama_guard_generation = _create_llama_guard_generation(training_example, category_indices_to_include_in_llama_guard_prompt, formatter_configs)
    return f'{llama_guard_prompt} {llama_guard_generation}'