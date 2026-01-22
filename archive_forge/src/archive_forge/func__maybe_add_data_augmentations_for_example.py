import copy
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional, Sequence
def _maybe_add_data_augmentations_for_example(training_example: TrainingExample, formatted_examples_being_built: List[str], indices_of_all_categories: range, formatter_configs: FormatterConfigs) -> None:
    violated_category_indices = _convert_category_codes_to_indices(training_example.violated_category_codes, formatter_configs)
    nonviolated_category_indices = list(set(indices_of_all_categories) - set(violated_category_indices))
    _maybe_add_example_with_dropped_nonviolated_prompt_categories(training_example, formatted_examples_being_built, indices_of_all_categories, nonviolated_category_indices, formatter_configs)
    _maybe_add_example_with_dropped_violated_and_nonviolated_prompt_categories(training_example, formatted_examples_being_built, indices_of_all_categories, violated_category_indices, nonviolated_category_indices, formatter_configs)