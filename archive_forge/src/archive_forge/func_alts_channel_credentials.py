import abc
import contextlib
import enum
import logging
import sys
from grpc import _compression
from grpc._cython import cygrpc as _cygrpc
from grpc._runtime_protos import protos
from grpc._runtime_protos import protos_and_services
from grpc._runtime_protos import services
def alts_channel_credentials(service_accounts=None):
    """Creates a ChannelCredentials for use with an ALTS-enabled Channel.

    This is an EXPERIMENTAL API.
    ALTS credentials API can only be used in GCP environment as it relies on
    handshaker service being available. For more info about ALTS see
    https://cloud.google.com/security/encryption-in-transit/application-layer-transport-security

    Args:
      service_accounts: A list of server identities accepted by the client.
        If target service accounts are provided and none of them matches the
        peer identity of the server, handshake will fail. The arg can be empty
        if the client does not have any information about trusted server
        identity.
    Returns:
      A ChannelCredentials for use with an ALTS-enabled Channel
    """
    return ChannelCredentials(_cygrpc.channel_credentials_alts(service_accounts or []))