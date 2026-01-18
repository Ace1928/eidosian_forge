from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import os
from googlecloudsdk.command_lib.kuberun import messages
from googlecloudsdk.command_lib.util.anthos import binary_operations
Binary operation wrapper for kuberun commands that require streaming output.