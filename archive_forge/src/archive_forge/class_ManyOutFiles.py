import os.path as op
import networkx as nx
from nipype import Workflow, MapNode, Node, IdentityInterface
from nipype.interfaces.base import (  # BaseInterfaceInputSpec,
class ManyOutFiles(TraitedSpec):
    out_files = OutputMultiPath(File(exists=True))