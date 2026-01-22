from typing import Dict, Union, Optional, TYPE_CHECKING
from ray.util.annotations import PublicAPI
@PublicAPI(stability='alpha')
class In:

    def __init__(self, *values):
        _validate_label_match_operator_values(values, 'In')
        self.values = list(values)