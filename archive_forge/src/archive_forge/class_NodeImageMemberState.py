from libcloud.common.types import (
class NodeImageMemberState(Type):
    """
    Standard states of VolumeSnapshots
    """
    ACCEPTED = 'accepted'
    PENDING = 'pending'
    REJECTED = 'rejected'