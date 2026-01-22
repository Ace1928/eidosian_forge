from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from functools import partial
from apitools.base.py import encoding
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import text
from sqlparse import lexer
from sqlparse import tokens as T
def _ConvertToTree(plan_nodes):
    """Creates tree of Node objects from the plan_nodes in server response.

  Args:
    plan_nodes (spanner_v1_messages.PlanNode[]): The plan_nodes from the server
      response. Plan nodes are topologically sorted.

  Returns:
    A Node, root of a tree built from `plan_nodes`.
  """
    return _BuildSubTree(plan_nodes, plan_nodes[0])