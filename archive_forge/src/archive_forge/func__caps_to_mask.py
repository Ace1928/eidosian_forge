import enum
import os
import platform
import sys
import cffi
def _caps_to_mask(caps):
    """Convert list of bit offsets to bitmask"""
    mask = 0
    for cap in caps:
        mask |= 1 << cap
    return mask