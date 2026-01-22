import os.path as op
import networkx as nx
from nipype import Workflow, MapNode, Node, IdentityInterface
from nipype.interfaces.base import (  # BaseInterfaceInputSpec,
def add_regexp_substitutions(self, sub_list):
    """Safely set subsitutions implemented with regular expressions."""
    if isdefined(self.sink_node.inputs.regexp_substitutions):
        self.sink_node.inputs.regexp_substitutions.extend(sub_list)
    else:
        self.sink_node.inputs.regexp_substitutions = sub_list