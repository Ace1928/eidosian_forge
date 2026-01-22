import torch
from typing import Any, Set, Dict, List, Tuple, OrderedDict
from collections import OrderedDict as OrdDict
def _get_filtered_data(self, feature_filter: str, module_fqn_filter: str) -> OrderedDict[str, Any]:
    """
        Filters the data and returns it in the same ordered dictionary format so the relevant views can be displayed.

        Args:
            feature_filter (str): The feature filter, if we want to filter the set of data to only include
                a certain set of features that include feature_filter
                If feature = "", then we do not filter based on any features
            module_fqn_filter (str): The filter on prefix for the module fqn. All modules that have fqn with
                this prefix will be included
                If module_fqn_filter = "" we do not filter based on module fqn, and include all modules

        First, the data is filtered based on module_fqn, and then filtered based on feature
        Returns an OrderedDict (sorted in order of model) mapping:
            module_fqns -> feature_names -> values
        """
    filtered_dict: OrderedDict[str, Any] = OrdDict()
    for module_fqn in self.generated_reports:
        if module_fqn_filter == '' or module_fqn_filter in module_fqn:
            filtered_dict[module_fqn] = {}
            module_reports = self.generated_reports[module_fqn]
            for feature_name in module_reports:
                if feature_filter == '' or feature_filter in feature_name:
                    filtered_dict[module_fqn][feature_name] = module_reports[feature_name]
    return filtered_dict