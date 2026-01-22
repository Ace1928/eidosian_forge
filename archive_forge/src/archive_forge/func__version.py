from base64 import b64encode
from typing import Mapping, Optional, NamedTuple
import logging
import pkg_resources
from cloudsdk.google.protobuf import struct_pb2  # pytype: disable=pyi-error
def _version() -> _Semver:
    try:
        version = pkg_resources.get_distribution('google-cloud-pubsublite').version
    except pkg_resources.DistributionNotFound:
        _LOGGER.info('Failed to extract the google-cloud-pubsublite semver version. DistributionNotFound.')
        return _Semver(0, 0)
    splits = version.split('.')
    if len(splits) != 3:
        _LOGGER.info(f'Failed to extract semver from {version}.')
        return _Semver(0, 0)
    return _Semver(int(splits[0]), int(splits[1]))