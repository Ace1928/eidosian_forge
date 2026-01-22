import copy
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional, Sequence
def _maybe_add_example_with_dropped_nonviolated_prompt_categories(training_example: TrainingExample, formatted_examples_being_built: List[str], indices_of_all_categories: range, nonviolated_category_indices: List[int], formatter_configs: FormatterConfigs) -> None:
    """
    If a prompt+response pair does not violate certain categories, we can augment
    the data by duplicating the training example but removing some of the non-violated
    categories from the llama guard prompt. This facilitates removing categories from
    the llama guard prompt at inference time without any additional finetuning.
    """
    if not formatter_configs.augmentation_configs.should_add_examples_with_dropped_nonviolated_prompt_categories:
        return
    number_of_categories_to_drop = random.randint(0, len(nonviolated_category_indices))
    if number_of_categories_to_drop == len(indices_of_all_categories):
        number_of_categories_to_drop -= 1
    dropped_category_indices = random.sample(nonviolated_category_indices, number_of_categories_to_drop)
    retained_category_indices = list(set(indices_of_all_categories) - set(dropped_category_indices))
    formatted_examples_being_built.append(_create_formatted_finetuning_example(training_example, formatter_configs, category_indices_to_include_in_llama_guard_prompt=retained_category_indices))