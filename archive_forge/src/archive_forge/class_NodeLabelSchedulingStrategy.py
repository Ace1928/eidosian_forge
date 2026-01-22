from typing import Dict, Union, Optional, TYPE_CHECKING
from ray.util.annotations import PublicAPI
@PublicAPI(stability='alpha')
class NodeLabelSchedulingStrategy:
    """Label based node affinity scheduling strategy

    scheduling_strategy=NodeLabelSchedulingStrategy({
          "region": In("us"),
          "gpu_type": Exists()
    })
    """

    def __init__(self, hard: LabelMatchExpressionsT, *, soft: LabelMatchExpressionsT=None):
        self.hard = _convert_map_to_expressions(hard, 'hard')
        self.soft = _convert_map_to_expressions(soft, 'soft')
        self._check_usage()

    def _check_usage(self):
        if not (self.hard or self.soft):
            raise ValueError('The `hard` and `soft` parameter of NodeLabelSchedulingStrategy cannot both be empty.')