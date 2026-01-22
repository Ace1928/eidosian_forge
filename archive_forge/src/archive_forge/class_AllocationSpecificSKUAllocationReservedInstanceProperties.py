from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllocationSpecificSKUAllocationReservedInstanceProperties(_messages.Message):
    """Properties of the SKU instances being reserved. Next ID: 9

  Enums:
    MaintenanceIntervalValueValuesEnum: Specifies the frequency of planned
      maintenance events. The accepted values are: `PERIODIC`.

  Fields:
    guestAccelerators: Specifies accelerator type and count.
    localSsds: Specifies amount of local ssd to reserve with each instance.
      The type of disk is local-ssd.
    locationHint: An opaque location hint used to place the allocation close
      to other resources. This field is for use by internal tools that use the
      public API.
    machineType: Specifies type of machine (name only) which has fixed number
      of vCPUs and fixed amount of memory. This also includes specifying
      custom machine type following custom-NUMBER_OF_CPUS-AMOUNT_OF_MEMORY
      pattern.
    maintenanceFreezeDurationHours: Specifies the number of hours after
      reservation creation where instances using the reservation won't be
      scheduled for maintenance.
    maintenanceInterval: Specifies the frequency of planned maintenance
      events. The accepted values are: `PERIODIC`.
    minCpuPlatform: Minimum cpu platform the reservation.
  """

    class MaintenanceIntervalValueValuesEnum(_messages.Enum):
        """Specifies the frequency of planned maintenance events. The accepted
    values are: `PERIODIC`.

    Values:
      AS_NEEDED: VMs are eligible to receive infrastructure and hypervisor
        updates as they become available. This may result in more maintenance
        operations (live migrations or terminations) for the VM than the
        PERIODIC and RECURRENT options.
      PERIODIC: VMs receive infrastructure and hypervisor updates on a
        periodic basis, minimizing the number of maintenance operations (live
        migrations or terminations) on an individual VM. This may mean a VM
        will take longer to receive an update than if it was configured for
        AS_NEEDED. Security updates will still be applied as soon as they are
        available.
      RECURRENT: VMs receive infrastructure and hypervisor updates on a
        periodic basis, minimizing the number of maintenance operations (live
        migrations or terminations) on an individual VM. This may mean a VM
        will take longer to receive an update than if it was configured for
        AS_NEEDED. Security updates will still be applied as soon as they are
        available. RECURRENT is used for GEN3 and Slice of Hardware VMs.
    """
        AS_NEEDED = 0
        PERIODIC = 1
        RECURRENT = 2
    guestAccelerators = _messages.MessageField('AcceleratorConfig', 1, repeated=True)
    localSsds = _messages.MessageField('AllocationSpecificSKUAllocationAllocatedInstancePropertiesReservedDisk', 2, repeated=True)
    locationHint = _messages.StringField(3)
    machineType = _messages.StringField(4)
    maintenanceFreezeDurationHours = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    maintenanceInterval = _messages.EnumField('MaintenanceIntervalValueValuesEnum', 6)
    minCpuPlatform = _messages.StringField(7)