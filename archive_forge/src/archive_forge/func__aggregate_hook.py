from typing import Any, Dict, List, Optional
import torch
from collections import defaultdict
from torch import nn
import copy
from ...sparsifier.utils import fqn_to_module, module_to_fqn
import warnings
def _aggregate_hook(self, name):
    """Returns hook that computes aggregate of activations passing through.
        """
    feature_dim = self.data_groups[name]['feature_dim']
    features = self.data_groups[name]['features']
    agg_fn = self.data_groups[name]['aggregate_fn']

    def hook(module, input) -> None:
        input_data = input[0]
        data = self.data_groups[name].get('data')
        if features is None:
            if data is None:
                data = torch.zeros_like(input_data)
                self.state[name]['mask'] = torch.ones_like(input_data)
            out_data = agg_fn(data, input_data)
        else:
            if data is None:
                out_data = [0 for _ in range(0, len(features))]
                self.state[name]['mask'] = [0 for _ in range(0, len(features))]
            else:
                out_data = data
            for feature_idx in range(len(features)):
                feature_tensor = torch.Tensor([features[feature_idx]]).long().to(input_data.device)
                data_feature = torch.index_select(input_data, feature_dim, feature_tensor)
                if data is None:
                    curr_data = torch.zeros_like(data_feature)
                    self.state[name]['mask'][feature_idx] = torch.ones_like(data_feature)
                else:
                    curr_data = data[feature_idx]
                out_data[feature_idx] = agg_fn(curr_data, data_feature)
        self.data_groups[name]['data'] = out_data
    return hook