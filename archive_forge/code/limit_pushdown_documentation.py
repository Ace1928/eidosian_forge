import copy
from collections import deque
from typing import Iterable, List
from ray.data._internal.logical.interfaces import LogicalOperator, LogicalPlan, Rule
from ray.data._internal.logical.operators.one_to_one_operator import (
from ray.data._internal.logical.operators.read_operator import Read
Given a DAG of LogicalOperators, traverse the DAG and fuse all
        back-to-back Limit operators, i.e.
        Limit[n] -> Limit[m] becomes Limit[min(n, m)].

        Returns a new LogicalOperator with the Limit operators fusion applied.