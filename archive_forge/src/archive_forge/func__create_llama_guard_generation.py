import copy
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional, Sequence
def _create_llama_guard_generation(training_example: TrainingExample, category_indices_included_in_llama_guard_prompt: List[int], formatter_configs: FormatterConfigs) -> str:
    to_return = training_example.label
    if training_example.label == 'unsafe' and formatter_configs.llama_guard_generation_configs.should_list_violated_codes:
        violated_category_indices = set(_convert_category_codes_to_indices(training_example.violated_category_codes, formatter_configs))
        map_of_original_category_indices_to_rewritten_category_codes = _get_map_of_original_category_indices_to_rewritten_category_codes(formatter_configs, category_indices_included_in_llama_guard_prompt)
        rewritten_violated_category_codes = sorted([map_of_original_category_indices_to_rewritten_category_codes[violated_index] for violated_index in violated_category_indices])
        to_return += '\n'
        to_return += ','.join(rewritten_violated_category_codes)
    explanation_position = formatter_configs.llama_guard_generation_configs.explanation_position
    if explanation_position == ExplanationPosition.BEFORE_DECISION:
        to_return = f'Explanation: {training_example.explanation}\n{to_return}'
    elif explanation_position == ExplanationPosition.AFTER_DECISION:
        to_return = f'{to_return}\nExplanation: {training_example.explanation}'
    return to_return