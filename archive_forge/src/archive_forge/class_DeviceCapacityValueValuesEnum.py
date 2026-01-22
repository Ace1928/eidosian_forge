from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DeviceCapacityValueValuesEnum(_messages.Enum):
    """The number of online devices for an iOS version.

    Values:
      DEVICE_CAPACITY_UNSPECIFIED: The value of device capacity is unknown or
        unset.
      DEVICE_CAPACITY_HIGH: Devices that are high in capacity (The lab has a
        large number of these devices). These devices are generally suggested
        for running a large number of simultaneous tests (e.g. more than 100
        tests). Please note that high capacity devices do not guarantee short
        wait times due to several factors: 1. Traffic (how heavily they are
        used at any given moment) 2. High capacity devices are prioritized for
        certain usages, which may cause user tests to be slower than selecting
        other similar device types.
      DEVICE_CAPACITY_MEDIUM: Devices that are medium in capacity (The lab has
        a decent number of these devices, though not as many as high capacity
        devices). These devices are suitable for fewer test runs (e.g. fewer
        than 100 tests) and only for low shard counts (e.g. less than 10
        shards).
      DEVICE_CAPACITY_LOW: Devices that are low in capacity (The lab has a
        small number of these devices). These devices may be used if users
        need to test on this specific device model and version. Please note
        that due to low capacity, the tests may take much longer to finish,
        especially if a large number of tests are invoked at once. These
        devices are not suitable for test sharding.
      DEVICE_CAPACITY_NONE: Devices that are completely missing from the lab.
        These devices are unavailable either temporarily or permanently and
        should not be requested. If the device is also marked as deprecated,
        this state is very likely permanent.
    """
    DEVICE_CAPACITY_UNSPECIFIED = 0
    DEVICE_CAPACITY_HIGH = 1
    DEVICE_CAPACITY_MEDIUM = 2
    DEVICE_CAPACITY_LOW = 3
    DEVICE_CAPACITY_NONE = 4