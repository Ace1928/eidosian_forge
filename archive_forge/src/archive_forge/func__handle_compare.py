import ast
import json
import sys
import urllib
from wandb_gql import gql
import wandb
from wandb.apis import public
from wandb.apis.attrs import Attrs
from wandb.apis.paginator import Paginator
from wandb.sdk.lib import ipython
def _handle_compare(self, node):
    left = self.front_to_back(self._handle_fields(node.left))
    op = self._handle_ops(node.ops[0])
    right = self._handle_fields(node.comparators[0])
    if op == '=':
        return {left: right}
    else:
        return {left: {op: right}}