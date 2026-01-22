from pprint import pformat
from six import iteritems
import re
@aggregation_rule.setter
def aggregation_rule(self, aggregation_rule):
    """
        Sets the aggregation_rule of this V1ClusterRole.
        AggregationRule is an optional field that describes how to build the
        Rules for this ClusterRole. If AggregationRule is set, then the Rules
        are controller managed and direct changes to Rules will be stomped by
        the controller.

        :param aggregation_rule: The aggregation_rule of this V1ClusterRole.
        :type: V1AggregationRule
        """
    self._aggregation_rule = aggregation_rule