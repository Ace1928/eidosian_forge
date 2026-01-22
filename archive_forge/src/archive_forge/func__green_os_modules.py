from __future__ import annotations
import sys
import eventlet
def _green_os_modules():
    from eventlet.green import os
    return [('os', os)]