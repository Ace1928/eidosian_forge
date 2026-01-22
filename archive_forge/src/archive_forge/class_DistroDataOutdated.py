import csv
import datetime
import os
class DistroDataOutdated(Exception):
    """Distribution data outdated."""

    def __init__(self):
        super().__init__('Distribution data outdated. Please check for an update for distro-info-data. See /usr/share/doc/distro-info-data/README.Debian for details.')