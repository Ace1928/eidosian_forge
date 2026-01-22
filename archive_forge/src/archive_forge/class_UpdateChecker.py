import os
import pickle
import re
import requests
import sys
import time
from datetime import datetime
from functools import wraps
from tempfile import gettempdir
class UpdateChecker:
    """A class to check for package updates."""

    def __init__(self, *, bypass_cache=False):
        self._bypass_cache = bypass_cache

    @cache_results
    def check(self, package_name, package_version):
        """Return a UpdateResult object if there is a newer version."""
        data = query_pypi(package_name, include_prereleases=not standard_release(package_version))
        if not data.get('success') or parse_version(package_version) >= parse_version(data['data']['version']):
            return None
        return UpdateResult(package_name, running=package_version, available=data['data']['version'], release_date=data['data']['upload_time'])