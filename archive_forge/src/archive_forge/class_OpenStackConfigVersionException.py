from openstack import exceptions
class OpenStackConfigVersionException(OpenStackConfigException):
    """A version was requested that is different than what was found."""

    def __init__(self, version):
        super(OpenStackConfigVersionException, self).__init__()
        self.version = version