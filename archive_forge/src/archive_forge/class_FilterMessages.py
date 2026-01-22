import re
import sys
import time
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
from docutils.utils import smartquotes
class FilterMessages(Transform):
    """
    Remove system messages below verbosity threshold.
    """
    default_priority = 870

    def apply(self):
        for node in self.document.traverse(nodes.system_message):
            if node['level'] < self.document.reporter.report_level:
                node.parent.remove(node)