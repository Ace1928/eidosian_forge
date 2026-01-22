import torch
from typing import Any, Set, Dict, List, Tuple, OrderedDict
from collections import OrderedDict as OrdDict
def _generate_tensor_table(self, filtered_data: OrderedDict[str, Dict[str, Any]], tensor_features: List[str]) -> Tuple[List, List]:
    """
        Takes in the filtered data and features list and generates the tensor headers and table

        Currently meant to generate the headers and table for both the tensor information.

        Args:
            filtered_data (OrderedDict[str, Dict[str, Any]]): An OrderedDict (sorted in order of model) mapping:
                module_fqns -> feature_names -> values
            tensor_features (List[str]): A list of the tensor level features

        Returns a tuple with:
            A list of the headers of the tensor table
            A list of lists containing the table information row by row
            The 0th index row will contain the headers of the columns
            The rest of the rows will contain data
        """
    tensor_table: List[List[Any]] = []
    tensor_headers: List[str] = []
    if len(tensor_features) > 0:
        for index, module_fqn in enumerate(filtered_data):
            tensor_table_row = [index, module_fqn]
            for feature in tensor_features:
                if feature in filtered_data[module_fqn]:
                    feature_val = filtered_data[module_fqn][feature]
                else:
                    feature_val = 'Not Applicable'
                if isinstance(feature_val, torch.Tensor):
                    feature_val = feature_val.item()
                tensor_table_row.append(feature_val)
            tensor_table.append(tensor_table_row)
    if len(tensor_table) != 0:
        tensor_headers = ['idx', 'layer_fqn'] + tensor_features
    return (tensor_headers, tensor_table)