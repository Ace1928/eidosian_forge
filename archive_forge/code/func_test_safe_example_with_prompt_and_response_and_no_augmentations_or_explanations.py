from enum import Enum
import unittest
from typing import Optional, List
from llama_recipes.data.llama_guard.finetuning_data_formatter import (
def test_safe_example_with_prompt_and_response_and_no_augmentations_or_explanations(self) -> None:
    formatter_configs = FinetuningDataFormatterTests.create_most_conservative_formatter_configs(AgentType.AGENT)
    formatter_configs.llama_guard_generation_configs.explanation_position = None
    self.format_examples_and_compare_to_expected([TrainingExample(prompt='Where can I buy a car?', response='At the car store.', violated_category_codes=[], label='safe')], ["[INST] Task: Check if there is unsafe content in 'AgentType.AGENT' messages in conversations according our safety policy with the below categories.\n<BEGIN UNSAFE CONTENT CATEGORIES>\nO1: cat V. \ncat V description\nO2: cat W. \ncat W description\nO3: cat X. \ncat X description\nO4: cat Y. \ncat Y description\nO5: cat Z. \ncat Z description\n<END UNSAFE CONTENT CATEGORIES>\n\n<BEGIN CONVERSATION>\n\nhuman: Where can I buy a car?\n\nchatbot: At the car store.\n\n<END CONVERSATION>\n\nProvide your safety assessment for AgentType.AGENT in the above conversation:\n- First line must read 'safe' or 'unsafe'.\n- If unsafe, a second line must include a comma-separated list of violated categories. [/INST] safe"], AgentType.AGENT, formatter_configs)