import abc
import ctypes
import inspect
import json
import warnings
from collections import OrderedDict
from copy import deepcopy
from enum import Enum
from functools import wraps
from os import SEEK_END, environ
from os.path import getsize
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
import scipy.sparse
from .compat import (PANDAS_INSTALLED, PYARROW_INSTALLED, arrow_cffi, arrow_is_floating, arrow_is_integer, concat,
from .libpath import find_lib_path
def trees_to_dataframe(self) -> pd_DataFrame:
    """Parse the fitted model and return in an easy-to-read pandas DataFrame.

        The returned DataFrame has the following columns.

            - ``tree_index`` : int64, which tree a node belongs to. 0-based, so a value of ``6``, for example, means "this node is in the 7th tree".
            - ``node_depth`` : int64, how far a node is from the root of the tree. The root node has a value of ``1``, its direct children are ``2``, etc.
            - ``node_index`` : str, unique identifier for a node.
            - ``left_child`` : str, ``node_index`` of the child node to the left of a split. ``None`` for leaf nodes.
            - ``right_child`` : str, ``node_index`` of the child node to the right of a split. ``None`` for leaf nodes.
            - ``parent_index`` : str, ``node_index`` of this node's parent. ``None`` for the root node.
            - ``split_feature`` : str, name of the feature used for splitting. ``None`` for leaf nodes.
            - ``split_gain`` : float64, gain from adding this split to the tree. ``NaN`` for leaf nodes.
            - ``threshold`` : float64, value of the feature used to decide which side of the split a record will go down. ``NaN`` for leaf nodes.
            - ``decision_type`` : str, logical operator describing how to compare a value to ``threshold``.
              For example, ``split_feature = "Column_10", threshold = 15, decision_type = "<="`` means that
              records where ``Column_10 <= 15`` follow the left side of the split, otherwise follows the right side of the split. ``None`` for leaf nodes.
            - ``missing_direction`` : str, split direction that missing values should go to. ``None`` for leaf nodes.
            - ``missing_type`` : str, describes what types of values are treated as missing.
            - ``value`` : float64, predicted value for this leaf node, multiplied by the learning rate.
            - ``weight`` : float64 or int64, sum of Hessian (second-order derivative of objective), summed over observations that fall in this node.
            - ``count`` : int64, number of records in the training data that fall into this node.

        Returns
        -------
        result : pandas DataFrame
            Returns a pandas DataFrame of the parsed model.
        """
    if not PANDAS_INSTALLED:
        raise LightGBMError('This method cannot be run without pandas installed. You must install pandas and restart your session to use this method.')
    if self.num_trees() == 0:
        raise LightGBMError('There are no trees in this Booster and thus nothing to parse')

    def _is_split_node(tree: Dict[str, Any]) -> bool:
        return 'split_index' in tree.keys()

    def create_node_record(tree: Dict[str, Any], node_depth: int=1, tree_index: Optional[int]=None, feature_names: Optional[List[str]]=None, parent_node: Optional[str]=None) -> Dict[str, Any]:

        def _get_node_index(tree: Dict[str, Any], tree_index: Optional[int]) -> str:
            tree_num = f'{tree_index}-' if tree_index is not None else ''
            is_split = _is_split_node(tree)
            node_type = 'S' if is_split else 'L'
            node_num = tree.get('split_index' if is_split else 'leaf_index', 0)
            return f'{tree_num}{node_type}{node_num}'

        def _get_split_feature(tree: Dict[str, Any], feature_names: Optional[List[str]]) -> Optional[str]:
            if _is_split_node(tree):
                if feature_names is not None:
                    feature_name = feature_names[tree['split_feature']]
                else:
                    feature_name = tree['split_feature']
            else:
                feature_name = None
            return feature_name

        def _is_single_node_tree(tree: Dict[str, Any]) -> bool:
            return set(tree.keys()) == {'leaf_value'}
        node: Dict[str, Union[int, str, None]] = OrderedDict()
        node['tree_index'] = tree_index
        node['node_depth'] = node_depth
        node['node_index'] = _get_node_index(tree, tree_index)
        node['left_child'] = None
        node['right_child'] = None
        node['parent_index'] = parent_node
        node['split_feature'] = _get_split_feature(tree, feature_names)
        node['split_gain'] = None
        node['threshold'] = None
        node['decision_type'] = None
        node['missing_direction'] = None
        node['missing_type'] = None
        node['value'] = None
        node['weight'] = None
        node['count'] = None
        if _is_split_node(tree):
            node['left_child'] = _get_node_index(tree['left_child'], tree_index)
            node['right_child'] = _get_node_index(tree['right_child'], tree_index)
            node['split_gain'] = tree['split_gain']
            node['threshold'] = tree['threshold']
            node['decision_type'] = tree['decision_type']
            node['missing_direction'] = 'left' if tree['default_left'] else 'right'
            node['missing_type'] = tree['missing_type']
            node['value'] = tree['internal_value']
            node['weight'] = tree['internal_weight']
            node['count'] = tree['internal_count']
        else:
            node['value'] = tree['leaf_value']
            if not _is_single_node_tree(tree):
                node['weight'] = tree['leaf_weight']
                node['count'] = tree['leaf_count']
        return node

    def tree_dict_to_node_list(tree: Dict[str, Any], node_depth: int=1, tree_index: Optional[int]=None, feature_names: Optional[List[str]]=None, parent_node: Optional[str]=None) -> List[Dict[str, Any]]:
        node = create_node_record(tree=tree, node_depth=node_depth, tree_index=tree_index, feature_names=feature_names, parent_node=parent_node)
        res = [node]
        if _is_split_node(tree):
            children = ['left_child', 'right_child']
            for child in children:
                subtree_list = tree_dict_to_node_list(tree=tree[child], node_depth=node_depth + 1, tree_index=tree_index, feature_names=feature_names, parent_node=node['node_index'])
                res.extend(subtree_list)
        return res
    model_dict = self.dump_model()
    feature_names = model_dict['feature_names']
    model_list = []
    for tree in model_dict['tree_info']:
        model_list.extend(tree_dict_to_node_list(tree=tree['tree_structure'], tree_index=tree['tree_index'], feature_names=feature_names))
    return pd_DataFrame(model_list, columns=model_list[0].keys())