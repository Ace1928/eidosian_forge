from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def MakePerInstanceConfigFromDiskAndMetadataDicts(messages, name, disks=None, metadata=None):
    """Create a per-instance config message from disks and metadata attributes.

  Args:
    messages: Messages module
    name: Name of the instance
    disks: list of disk dictionaries, eg. [{
          'device_name': 'foo',
          'source': '../projects/project-foo/.../disks/disk-a',
          'auto_delete': 'on-permanent-instance-deletion' }]
    metadata: list of metadata dictionaries, eg. [{
          'key': 'my-key',
          'value': 'my-value', }]

  Returns:
    per-instance config message
  """
    preserved_state_disks = []
    for disk_dict in disks or []:
        preserved_state_disks.append(MakePreservedStateDiskMapEntry(messages, *disk_dict))
    preserved_state_metadata = []
    for metadata_dict in metadata or []:
        preserved_state_metadata.append(MakePreservedStateMetadataMapEntry(messages, *metadata_dict))
    return MakePerInstanceConfig(messages, name, preserved_state_disks, preserved_state_metadata)