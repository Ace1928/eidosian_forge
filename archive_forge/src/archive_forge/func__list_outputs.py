import os.path as op
import networkx as nx
from nipype import Workflow, MapNode, Node, IdentityInterface
from nipype.interfaces.base import (  # BaseInterfaceInputSpec,
def _list_outputs(self):
    outputs = self._outputs().get()
    outputs['out_file'] = op.abspath(fname)
    return outputs