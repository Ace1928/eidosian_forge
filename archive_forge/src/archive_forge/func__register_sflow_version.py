import struct
import logging
def _register_sflow_version(cls):
    sFlow._SFLOW_VERSIONS[version] = cls
    return cls