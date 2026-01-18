import logging
from pyVim.task import WaitForTask
from pyVmomi import vim

    This function helps to add a list of gpu to a VM by PCI passthrough. Steps:
    1. Power off the VM if it is not at the off state.
    2. Construct a reconfigure spec and reconfigure the VM.
    3. Power on the VM.
    