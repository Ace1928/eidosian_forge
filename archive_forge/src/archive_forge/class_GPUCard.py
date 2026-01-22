import logging
from pyVim.task import WaitForTask
from pyVmomi import vim
class GPUCard:

    def __init__(self, pci_id, custom_label=''):
        self.pciId = pci_id
        self.customLabel = custom_label

    def __str__(self):
        return 'pciId: %s, customLabel: %s' % (self.pciId, self.customLabel)

    def __repr__(self):
        return 'pciId: %s, customLabel: %s' % (self.pciId, self.customLabel)

    def __eq__(self, other):
        return self.pciId == other.pciId and self.customLabel == other.customLabel