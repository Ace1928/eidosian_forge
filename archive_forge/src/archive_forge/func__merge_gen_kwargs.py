from typing import List
import numpy as np
def _merge_gen_kwargs(gen_kwargs_list: List[dict]) -> dict:
    return {key: [value for gen_kwargs in gen_kwargs_list for value in gen_kwargs[key]] if isinstance(gen_kwargs_list[0][key], list) else gen_kwargs_list[0][key] for key in gen_kwargs_list[0]}