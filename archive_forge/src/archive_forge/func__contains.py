from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
def _contains(potential_container_url, potential_containee_url):
    """Checks containment based on string representations."""
    potential_container_string = potential_container_url.versionless_url_string
    potential_containee_string = potential_containee_url.versionless_url_string
    delimiter = potential_container_url.delimiter
    prefix = potential_container_string.rstrip(delimiter) + delimiter
    return potential_containee_string.startswith(prefix)