from openstack.block_storage.v3 import volume as _volume
from openstack.tests.functional.block_storage.v3 import base
Test class for volume attachment operations.

    We have implemented a test that performs attachment create
    and attachment delete operations. Attachment create requires
    the instance ID and the volume ID for which we have created a
    volume resource and an instance resource.
    We haven't implemented attachment update test since it requires
    the host connector information which is not readily available to
    us and hard to retrieve. Without passing this information, the
    attachment update operation will fail.
    Similarly, we haven't implement attachment complete test since it
    depends on attachment update and can only be performed when the volume
    status is 'attaching' which is done by attachment update operation.
    