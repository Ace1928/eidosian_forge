import pytest
from flaky import flaky
import pennylane as qml
from pennylane import numpy as pnp  # Import from PennyLane to mirror the standard approach in demos
from pennylane.templates.layers import RandomLayers
def workload():
    return (qnode(theta, phi, state), qnode_def(theta, phi, state), grad(theta, phi, state), grad_def(theta, phi, state))