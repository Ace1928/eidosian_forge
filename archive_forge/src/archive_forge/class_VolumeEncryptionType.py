from cinderclient.apiclient import base as common_base
from cinderclient import base
class VolumeEncryptionType(base.Resource):
    """
    A Volume Encryption Type is a collection of settings used to conduct
    encryption for a specific volume type.
    """

    def __repr__(self):
        return '<VolumeEncryptionType: %s>' % self.encryption_id