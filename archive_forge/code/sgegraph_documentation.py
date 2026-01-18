import os
import sys
from ...interfaces.base import CommandLine
from .base import GraphPluginBase, logger

            - jobnumber: The index number of the job to create
            - nodeslist: The name of the node being processed
            - return: A string representing this job to be displayed by SGE
            