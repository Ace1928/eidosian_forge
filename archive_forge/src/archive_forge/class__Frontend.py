from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import filter_scope_rewriter
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_expr_rewrite
from googlecloudsdk.core.resource import resource_projector
import six
class _Frontend(object):
    """Example of conforming Frontend implementation."""

    def __init__(self, filter_expr=None, maxResults=None, scopeSet=None):
        self._filter_expr = filter_expr
        self._max_results = maxResults
        self._scope_set = scopeSet

    @property
    def filter(self):
        return self._filter_expr

    @property
    def max_results(self):
        return self._max_results

    @property
    def scope_set(self):
        return self._scope_set