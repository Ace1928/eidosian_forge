import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def _match_images(self, project, partial_name):
    """
        Find the latest image, given a partial name.

        For example, providing 'debian-7' will return the image object for the
        most recent image with a name that starts with 'debian-7' in the
        supplied project.  If no project is given, it will search your own
        project.

        :param  project: The name of the project to search for images.
                         Examples include: 'debian-cloud' and 'centos-cloud'.
        :type   project: ``str``, ``list`` of ``str``, or ``None``

        :param  partial_name: The full name or beginning of a name for an
                              image.
        :type   partial_name: ``str``

        :return:  The latest image object that matches the partial name or None
                  if no matching image is found.
        :rtype:   :class:`GCENodeImage` or ``None``
        """
    project_images_list = self.ex_list(self.list_images, ex_project=project, ex_include_deprecated=True)
    partial_match = []
    for page in project_images_list.page():
        for image in page:
            if image.name == partial_name:
                return image
            if image.name.startswith(partial_name):
                ts = timestamp_to_datetime(image.extra['creationTimestamp'])
                if not partial_match or partial_match[0] < ts:
                    partial_match = [ts, image]
    if partial_match:
        return partial_match[1]