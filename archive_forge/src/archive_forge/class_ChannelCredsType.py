import enum
import functools
import random
import threading
import time
import grpc
from tensorboard import version
from tensorboard.util import tb_logging
@enum.unique
class ChannelCredsType(enum.Enum):
    LOCAL = 'local'
    SSL = 'ssl'
    SSL_DEV = 'ssl_dev'

    def channel_config(self):
        """Create channel credentials and options.

        Returns:
          A tuple `(channel_creds, channel_options)`, where `channel_creds`
          is a `grpc.ChannelCredentials` and `channel_options` is a
          (potentially empty) list of `(key, value)` tuples. Both results
          may be passed to `grpc.secure_channel`.
        """
        options = []
        if self == ChannelCredsType.LOCAL:
            creds = grpc.local_channel_credentials()
        elif self == ChannelCredsType.SSL:
            creds = grpc.ssl_channel_credentials()
        elif self == ChannelCredsType.SSL_DEV:
            creds = grpc.ssl_channel_credentials()
            options.append(('grpc.ssl_target_name_override', 'localhost'))
        else:
            raise AssertionError('unhandled ChannelCredsType: %r' % self)
        return (creds, options)

    @classmethod
    def choices(cls):
        return cls.__members__.values()

    def __str__(self):
        return self.value